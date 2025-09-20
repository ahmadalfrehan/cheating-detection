import pyopenpose as op
import os
import sys
import cv2
import numpy as np
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import pyaudio
import audioop

# Add DLL paths before import
os.add_dll_directory(r"C:/openpose/build/x64/Release")
os.add_dll_directory(
    r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
sys.path.append(r"C:/openpose/build/python/openpose/Release")

# Parameters
params = {
    "model_folder": "C:/openpose/models",
    "model_pose": "BODY_25",
    "net_resolution": "-1x160",
    "disable_blending": False,
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Video & YOLO
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Constants
THRESHOLD = 1.1
COOLDOWN_PERIOD = timedelta(seconds=2)
NOISE_THRESHOLD = 100
HISTORY_LENGTH = 10
PROCESS_EVERY_SECONDS = 0.3

ANGLE_THRESHOLD = 20
MAX_BASELINE_ALLOWED = 15
MIN_DEVIATION_FROM_BASELINE = 6
MIN_HOLD_TIME = 0.5
SYM_RATIO_MIN = 0.7
SYM_RATIO_MAX = 1.3

# NEW: Getting down detection parameters
DOWN_THRESHOLD = 40  # Minimum downward movement to trigger alert
DOWN_HOLD_TIME = 0.3  # How long person must be down to trigger alert

# Globals
tracking_data = {}
scoreboard = defaultdict(int)
prev_people = []
prev_ids = []
tracked_persons = {}
person_id_counter = 1
last_alert_time = defaultdict(lambda: {"hand": None, "head": None, "stand": None, "down": None})
screenshot_count = 0
frame_count = 0
last_process_time = 0
noise_alert_triggered = False
save_dir = "alerts"
os.makedirs(save_dir, exist_ok=True)

# Listen for noise
def listen_for_noise():
    global noise_alert_triggered
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1,
                     rate=16000, input=True, frames_per_buffer=1024)
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        rms = audioop.rms(data, 2)
        if rms > NOISE_THRESHOLD:
            noise_alert_triggered = True

threading.Thread(target=listen_for_noise, daemon=True).start()

# Utils
def get_body_scale(person):
    if person[5][2] > 0.4 and person[2][2] > 0.4:
        return np.linalg.norm(person[5][:2] - person[2][:2])
    return 1.0

def update_history(pid, joint, pt):
    if pid not in tracking_data:
        tracking_data[pid] = {
            "left_hand": deque(maxlen=HISTORY_LENGTH), 
            "right_hand": deque(maxlen=HISTORY_LENGTH), 
            "left_shoulder_y": deque(maxlen=HISTORY_LENGTH), 
            "right_shoulder_y": deque(maxlen=HISTORY_LENGTH),
            "head_y": deque(maxlen=HISTORY_LENGTH),  # NEW: Track head height
            "torso_y": deque(maxlen=HISTORY_LENGTH)  # NEW: Track torso height
        }
    tracking_data[pid][joint].append(pt)

def compute_smoothed(pid, joint, pt):
    history = tracking_data[pid][joint]
    if not history:
        return 0
    avg = np.mean(history, axis=0)
    return np.linalg.norm(pt - avg)

def crop_and_save_alert(frame, person, pid, reason):
    global screenshot_count

    # Check if person is a valid list of keypoints
    if isinstance(person, np.ndarray) and person.ndim == 2 and person.shape[1] == 3:
        valid_points = [p[:2] for p in person if p[2] > 0.1]
        if not valid_points:
            cropped = frame.copy()  # Fall back to full frame if no keypoints
        else:
            points = np.array(valid_points)
            x, y, w, h = cv2.boundingRect(points.astype(np.int32))
            pad = 30
            x, y = max(x - pad, 0), max(y - pad, 0)
            x2, y2 = min(x + w + 2 * pad,
                         frame.shape[1]), min(y + h + 2 * pad, frame.shape[0])
            cropped = frame[y:y2, x:x2]
    else:
        # No keypoints, use full frame
        cropped = frame.copy()

    filename = os.path.join(
        save_dir, f"alert_{pid}_{reason}_{screenshot_count}.jpg")
    cv2.imwrite(filename, cropped)
    print(f"⚠️ {pid} {reason} alert -> {filename}")
    screenshot_count += 1

def assign_ids(current, previous, previous_ids, threshold=100):
    global person_id_counter
    new_ids = [None] * len(current)
    used = set()
    for i, p in enumerate(current):
        best_dist = float("inf")
        match = -1
        for j, q in enumerate(previous):
            if j in used:
                continue
            if p[1][2] < 0.1 or q[1][2] < 0.1:
                continue
            dist = np.linalg.norm(p[1][:2] - q[1][:2])
            if dist < best_dist:
                best_dist = dist
                match = j
        if match != -1 and best_dist < threshold:
            new_ids[i] = previous_ids[match]
            used.add(match)
        else:
            new_ids[i] = f"ID_{person_id_counter}"
            person_id_counter += 1
    return new_ids

# Main Loop
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    output_frame = frame.copy()
    current_time = time.time()

    if current_time - last_process_time >= PROCESS_EVERY_SECONDS:
        last_process_time = current_time

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        keypoints = datum.poseKeypoints
        output_frame = datum.cvOutputData.copy()
        filtered_keypoints = []
        if keypoints is not None:
            for person in keypoints:
                visible_joints = sum(1 for kp in person if kp[2] > 0.2)
                if visible_joints >= 4:
                    filtered_keypoints.append(person)
            keypoints = np.array(
                filtered_keypoints) if filtered_keypoints else None
        else:
            keypoints = None

        if keypoints is not None and keypoints.shape[0] > 0:
            ids = assign_ids(keypoints, prev_people, prev_ids)
            prev_people, prev_ids = keypoints.copy(), ids.copy()

            for i, person in enumerate(keypoints):
                pid = ids[i]
                scale = get_body_scale(person)
                hand_centers = []
                now = datetime.now()

                # === Hand movement ===
                if pid in tracked_persons and tracked_persons[pid]["excluded"]:
                    continue  # Skip hand detection for excluded people

                for joint_name, joint in {"left_hand": person[7], "right_hand": person[4]}.items():
                    if joint[2] < 0.4:
                        continue
                    update_history(pid, joint_name, joint[:2])
                    norm = compute_smoothed(pid, joint_name, joint[:2]) / scale
                    hand_centers.append(joint[:2])

                    if last_alert_time[pid]["hand"] and now - last_alert_time[pid]["hand"] < COOLDOWN_PERIOD:
                        continue

                    if norm >= THRESHOLD:
                        crop_and_save_alert(frame, person, pid, "hand")
                        scoreboard[pid] += 1
                        last_alert_time[pid]["hand"] = now

                # === Shoulder movement (standing detection) ===
                left_shoulder = person[5]
                right_shoulder = person[2]

                # Ensure shoulder and hand history keys are initialized
                if pid not in tracking_data:
                    tracking_data[pid] = {
                        "left_hand": deque(maxlen=HISTORY_LENGTH),
                        "right_hand": deque(maxlen=HISTORY_LENGTH),
                        "left_shoulder_y": deque(maxlen=HISTORY_LENGTH),
                        "right_shoulder_y": deque(maxlen=HISTORY_LENGTH),
                        "head_y": deque(maxlen=HISTORY_LENGTH),
                        "torso_y": deque(maxlen=HISTORY_LENGTH)
                    }
                else:
                    for key in ["left_shoulder_y", "right_shoulder_y", "head_y", "torso_y"]:
                        if key not in tracking_data[pid]:
                            tracking_data[pid][key] = deque(maxlen=HISTORY_LENGTH)

                if left_shoulder[2] > 0.4:
                    tracking_data[pid]["left_shoulder_y"].append(left_shoulder[1])
                if right_shoulder[2] > 0.4:
                    tracking_data[pid]["right_shoulder_y"].append(right_shoulder[1])

                # === NEW: Getting down detection ===
                nose = person[0]  # Head reference point
                neck = person[1]  # Neck/upper torso reference
                
                # Use the most reliable head/upper body reference point
                head_point = None
                if nose[2] > 0.4:
                    head_point = nose
                elif neck[2] > 0.4:
                    head_point = neck
                
                if head_point is not None:
                    # Track head height
                    tracking_data[pid]["head_y"].append(head_point[1])
                    
                    # Calculate torso center for better stability
                    torso_points = []
                    if left_shoulder[2] > 0.3:
                        torso_points.append(left_shoulder[1])
                    if right_shoulder[2] > 0.3:
                        torso_points.append(right_shoulder[1])
                    if neck[2] > 0.3:
                        torso_points.append(neck[1])
                    
                    if torso_points:
                        torso_center_y = np.mean(torso_points)
                        tracking_data[pid]["torso_y"].append(torso_center_y)
                
                # Check for getting down movement
                if (len(tracking_data[pid]["head_y"]) == HISTORY_LENGTH and 
                    len(tracking_data[pid]["torso_y"]) == HISTORY_LENGTH):
                    
                    # Calculate baseline positions (average of first few frames)
                    baseline_head = np.mean(list(tracking_data[pid]["head_y"])[:5])
                    baseline_torso = np.mean(list(tracking_data[pid]["torso_y"])[:5])
                    
                    # Current positions
                    current_head = head_point[1]
                    current_torso = torso_center_y if torso_points else current_head
                    
                    # Calculate downward movement (positive = moved down)
                    head_drop = current_head - baseline_head
                    torso_drop = current_torso - baseline_torso
                    
                    # Use the more significant drop
                    max_drop = max(head_drop, torso_drop)
                    
                    # Initialize down tracking if not exists
                    if pid not in tracked_persons:
                        tracked_persons[pid] = {
                            "disappear_start_time": None,
                            "disappeared": False,
                            "nose": tuple(nose[:2]),
                            "baseline_angle": None,
                            "turn_start_time": None,
                            "alerted": False,
                            "excluded": False,
                            "down_start_time": None,  # NEW
                            "down_alerted": False     # NEW
                        }
                    
                    # Add down tracking fields if missing
                    if "down_start_time" not in tracked_persons[pid]:
                        tracked_persons[pid]["down_start_time"] = None
                        tracked_persons[pid]["down_alerted"] = False
                    
                    # Check if person is significantly down
                    if max_drop > DOWN_THRESHOLD:
                        if tracked_persons[pid]["down_start_time"] is None:
                            tracked_persons[pid]["down_start_time"] = current_time
                        elif (current_time - tracked_persons[pid]["down_start_time"] >= DOWN_HOLD_TIME and 
                              not tracked_persons[pid]["down_alerted"]):
                            # Check cooldown
                            if (not last_alert_time[pid]["down"] or 
                                now - last_alert_time[pid]["down"] >= COOLDOWN_PERIOD):
                                crop_and_save_alert(frame, person, pid, "getting_down")
                                scoreboard[pid] += 1
                                last_alert_time[pid]["down"] = now
                                tracked_persons[pid]["down_alerted"] = True
                                print(f"🔽 {pid} getting down detected (drop: {max_drop:.1f}px)")
                    else:
                        # Reset down tracking if person is back up
                        tracked_persons[pid]["down_start_time"] = None
                        tracked_persons[pid]["down_alerted"] = False

                # === Standing detection (existing code) ===
                if (len(tracking_data[pid]["left_shoulder_y"]) == HISTORY_LENGTH and
                    len(tracking_data[pid]["right_shoulder_y"]) == HISTORY_LENGTH):
                    
                    avg_left = np.mean(tracking_data[pid]["left_shoulder_y"])
                    avg_right = np.mean(tracking_data[pid]["right_shoulder_y"])
                    base_height = max(avg_left, avg_right)

                    current_left = left_shoulder[1]
                    current_right = right_shoulder[1]
                    current_height = max(current_left, current_right)

                    rise = base_height - current_height  # positive if person moved up

                    if rise > 30:
                        if not last_alert_time[pid].get("stand") or now - last_alert_time[pid]["stand"] > COOLDOWN_PERIOD:
                            crop_and_save_alert(frame, person, pid, "standing")
                            scoreboard[pid] += 1
                            last_alert_time[pid]["stand"] = now

                # === Head turn detection (existing code) ===
                if nose[2] <= 0.1:
                    continue

                if pid not in tracked_persons:
                    tracked_persons[pid] = {
                        "disappear_start_time": None,
                        "disappeared": False,
                        "nose": tuple(nose[:2]),
                        "baseline_angle": None,
                        "turn_start_time": None,
                        "alerted": False,
                        "excluded": False,
                        "down_start_time": None,
                        "down_alerted": False
                    }
                elif "disappear_start_time" not in tracked_persons[pid]:
                    tracked_persons[pid]["disappear_start_time"] = None
                    tracked_persons[pid]["disappeared"] = False
                    
                shoulder_visible = left_shoulder[2] > 0.1 or right_shoulder[2] > 0.1
                data = tracked_persons[pid]
                
                if not shoulder_visible:
                    if data["disappear_start_time"] is None:
                        data["disappear_start_time"] = time.time()
                    elif not data["disappeared"] and time.time() - data["disappear_start_time"] > 3:
                        crop_and_save_alert(frame, person, pid, "disappear")
                        scoreboard[pid] += 1
                        data["disappeared"] = True
                else:
                    data["disappear_start_time"] = None
                    data["disappeared"] = False
                    
                data["nose"] = tuple(nose[:2])
                if data["excluded"]:
                    continue

                # Get facial references
                left_ear, right_ear = person[17], person[18]
                left_eye, right_eye = person[15], person[16]
                ref_left = left_ear[0] if left_ear[2] > 0.1 else left_eye[0] if left_eye[2] > 0.1 else None
                ref_right = right_ear[0] if right_ear[2] > 0.1 else right_eye[0] if right_eye[2] > 0.1 else None
                if ref_left is None or ref_right is None:
                    continue

                # Calculate face center and relative head angle
                center_x = (ref_left + ref_right) / 2
                offset = nose[0] - center_x
                face_width = abs(ref_left - ref_right)
                if face_width == 0:
                    continue

                angle = (offset / face_width) * 60

                # Baseline filtering
                if data["baseline_angle"] is None:
                    if abs(angle) > MAX_BASELINE_ALLOWED:
                        data["excluded"] = True
                        continue
                    data["baseline_angle"] = angle

                rel_angle = angle - data["baseline_angle"]

                # Wall-facing skip logic
                x_pos = nose[0]
                frame_width = frame.shape[1]
                edge_margin = 0.15 * frame_width

                if x_pos < edge_margin and rel_angle > 0:
                    continue
                if x_pos > (frame_width - edge_margin) and rel_angle < 0:
                    continue

                # Symmetry check
                symmetry_ratio = None
                try:
                    symmetry_ratio = abs(left_ear[0] - nose[0]) / abs(right_ear[0] - nose[0])
                except:
                    pass
                sym_ok = symmetry_ratio is None or symmetry_ratio < SYM_RATIO_MIN or symmetry_ratio > SYM_RATIO_MAX

                head_turn = (
                    abs(rel_angle) > ANGLE_THRESHOLD and
                    abs(rel_angle) > MIN_DEVIATION_FROM_BASELINE and
                    sym_ok
                )

                # Trigger head movement alert
                if head_turn:
                    if data["turn_start_time"] is None:
                        data["turn_start_time"] = current_time
                    elif current_time - data["turn_start_time"] >= MIN_HOLD_TIME and not data["alerted"]:
                        if not last_alert_time[pid]["head"] or now - last_alert_time[pid]["head"] >= COOLDOWN_PERIOD:
                            scoreboard[pid] += 1
                            crop_and_save_alert(frame, person, pid, "head")
                            last_alert_time[pid]["head"] = now
                            data["alerted"] = True
                else:
                    data["turn_start_time"] = None
                    data["alerted"] = False
        else:
            # If no one is detected in the frame, mark all currently tracked people as possibly disappeared
            now = time.time()
            for pid, data in tracked_persons.items():
                if data["disappear_start_time"] is None:
                    data["disappear_start_time"] = now
                elif not data["disappeared"] and now - data["disappear_start_time"] > 3:
                    print(f"🚨 {pid} disappeared (no keypoints at all)")
                    crop_and_save_alert(frame, frame, pid, "disappear")
                    scoreboard[pid] += 1
                    data["disappeared"] = True

    if noise_alert_triggered:
        cv2.putText(output_frame, "🚨 Noise Detected!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        noise_alert_triggered = False

    cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Cheating Detection - Enhanced with Getting Down", output_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if scoreboard:
    print("\n📊 FINAL CHEATING REPORT:")
    for student, score in scoreboard.items():
        print(f"🧑 {student}: {score} cheating points")
else:
    print("\n📊 No cheating detected during the session.")