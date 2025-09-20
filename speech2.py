import socketio

sio = socketio.Client()

@sio.event
def connect():
    print('✅ Connected to server')

@sio.event
def disconnect():
    print('❌ Disconnected from server')

@sio.on('hand_alert')
def handle_alert(data):
    print('⚠️ Hand Alert Received:', data)

sio.connect('http://127.0.0.1:5000')  # Replace with your IP and port
sio.wait()
