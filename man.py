# man.py - WebSocket manager for broadcasting alerts

import asyncio
from asyncio import subprocess
import subprocess
import base64
import json
import time
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
from typing import List, Dict
import logging
import threading

from fastapi.middleware.cors import CORSMiddleware
# Configure logging

exam_process = None 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (alert screenshots)
app.mount("/alerts", StaticFiles(directory="alerts"), name="alerts")


latest_frame = None
frame_lock = threading.Lock()
frame_timestamp = 0



class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = [] 
        self.client_info: Dict[WebSocket, Dict] = {}
        self.video_clients: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_info[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": asyncio.get_event_loop().time()
        }
        logger.info(f"Client connected: {self.client_info[websocket]['client_id']}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection",
            "message": "Connected to cheating detection system",
            "client_id": self.client_info[websocket]['client_id']
        }, websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            client_info = self.client_info.get(websocket, {})
            self.active_connections.remove(websocket)
            if websocket in self.client_info:
                del self.client_info[websocket]
            logger.info(f"Client disconnected: {client_info.get('client_id', 'unknown')}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            logger.warning("No active connections to broadcast to")
            return
            
        logger.info(f"Broadcasting to {len(self.active_connections)} clients: {message}")
        
        # Create a copy of connections to avoid modification during iteration
        connections_copy = self.active_connections.copy()
        
        for connection in connections_copy:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                self.disconnect(connection)

    async def broadcast_frame(self, frame_data: str):
        """Broadcast video frame to video clients only"""
        if not self.video_clients:
            return
            
        message = {
            "type": "video_frame",
            "data": frame_data,
            "timestamp": time.time()
            
        }
        
        # Create a copy to avoid modification during iteration
        video_clients_copy = self.video_clients.copy()
        
        for connection in video_clients_copy:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting frame to video client: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client messages if needed
                if message.get("type") == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": asyncio.get_event_loop().time()
                    }, websocket)

                elif message.get("type") == "start_exam":
                    # Trigger your OpenPose detection logic here
                    global exam_process
                    if exam_process is None or exam_process.poll() is not None:
                            # Start lastrules.py in a separate process
                            exam_process = subprocess.Popen(
                                ["python", "lastrules.py"], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            await manager.send_personal_message({
                                "type": "exam_status",
                                "status": "started",
                                "message": "lastrules.py started"
                            }, websocket)
                    else:
                            await manager.send_personal_message({
                                "type": "exam_status",
                                "status": "already_running"
                            }, websocket)

                elif message.get("type") == "stop_exam":
                    # global exam_process 
                    if exam_process is not None and exam_process.poll() is None:
                        exam_process.terminate()  # Stop script
                        exam_process = None
                        await manager.send_personal_message({
                            "type": "exam_status",
                            "status": "stopped",
                            "message": "lastrules.py terminated"
                        }, websocket)
                    else:
                        await manager.send_personal_message({
                            "type": "exam_status",
                            "status": "not_running"
                        }, websocket)
                        
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/video")
async def video_websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": time.time()
                    }, websocket)
                elif message.get("type") == "request_frame":
                    # Send current frame if available
                    if latest_frame is not None:
                        with frame_lock:
                            frame_copy = latest_frame.copy()
                        
                        # Encode frame as base64
                        _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        await manager.send_personal_message({
                            "type": "video_frame",
                            "data": frame_base64,
                            "timestamp": time.time()
                        }, websocket)
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Video WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/video_feed")
async def video_feed():
    """HTTP video streaming endpoint (alternative to WebSocket)"""
    def generate():
        while True:
            if latest_frame is not None:
                with frame_lock:
                    frame_copy = latest_frame.copy()
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def get_index():
    return {"message": "Cheating Detection WebSocket Server", "connections": len(manager.active_connections)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "clients": [info["client_id"] for info in manager.client_info.values()]
    }

# Video frame update function (called from main detection script)
def update_video_frame(frame):
    """Update the latest video frame for streaming"""
    global latest_frame, frame_timestamp
    with frame_lock:
        latest_frame = frame.copy()
        frame_timestamp = time.time()

# Async function to broadcast video frames periodically
async def video_broadcast_loop():
    """Continuously broadcast video frames to connected video clients"""
    while True:
        if latest_frame is not None and manager.video_clients:
            with frame_lock:
                frame_copy = latest_frame.copy()
            
            try:
                # Encode frame as base64 JPEG
                _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                await manager.broadcast_frame(frame_base64)
            except Exception as e:
                logger.error(f"Error in video broadcast loop: {e}")
        
        await asyncio.sleep(0.033)  # ~30 FPS

# Start video broadcasting task
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(video_broadcast_loop())

# Add startup event
@app.on_event("startup")
async def startup():
    await startup_event()

# Function to be called from your main detection script
async def broadcast_alert(alert_data: dict):
    """
    Broadcast alert to all connected clients
    
    Args:
        alert_data: Dictionary containing alert information
            - pid: Person ID
            - reason: Reason for alert (hand_movement, standing, etc.)
            - decision: Current decision status
            - screenshot_url: URL to screenshot
            - timestamp: Optional timestamp
    """
    try:
        # Add timestamp if not provided
        if "timestamp" not in alert_data:
            alert_data["timestamp"] = asyncio.get_event_loop().time()
        
        # Add alert type
        alert_data["type"] = "alert"
        
        # Broadcast to all connected clients
        await manager.broadcast(alert_data)
        
    except Exception as e:
        logger.error(f"Error broadcasting alert: {e}")

# Function to broadcast system status updates
async def broadcast_status(status_data: dict):
    """
    Broadcast system status updates
    
    Args:
        status_data: Dictionary containing status information
    """
    try:
        status_data["type"] = "status"
        status_data["timestamp"] = asyncio.get_event_loop().time()
        await manager.broadcast(status_data)
    except Exception as e:
        logger.error(f"Error broadcasting status: {e}")

# Function to broadcast scoreboard updates
async def broadcast_scoreboard(scoreboard: dict):
    """
    Broadcast current scoreboard to all clients
    
    Args:
        scoreboard: Dictionary of person_id -> score
    """
    try:
        message = {
            "type": "scoreboard",
            "data": scoreboard,
            "timestamp": asyncio.get_event_loop().time()
        }
        await manager.broadcast(message)
    except Exception as e:
        logger.error(f"Error broadcasting scoreboard: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")




# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.responses import StreamingResponse
# from fastapi.staticfiles import StaticFiles
# import asyncio

# app = FastAPI()
# clients = []

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     clients.append(websocket)
#     try:
#         while True:
#             await websocket.receive_text()
#     except WebSocketDisconnect:
#         clients.remove(websocket)

# @app.get("/video_feed")
# def video_feed():
#     from lastrules import generate_frames
#     return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# app.mount("/alerts", StaticFiles(directory="alerts"), name="alerts")

# async def broadcast_alert(alert_data: dict):
#     for ws in clients:
#         try:
#             await ws.send_json(alert_data)
#         except:
#             pass
