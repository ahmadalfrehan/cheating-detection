import os
import sys
import cv2
import numpy as np
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
from ultralytics import YOLO

import pyaudio
import audioop

# Add DLL paths before import
os.add_dll_directory(r"C:/openpose/build/x64/Release")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
sys.path.append(r"C:/openpose/build/python/openpose/Release")

from sys import platform
# try:
    
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


# Parameters
params = {
    "model_folder": "C:/openpose/models",
    "model_pose": "BODY_25",
    "net_resolution": "-1x160",
    "disable_blending": False,

}
params["logging_level"] = 3

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
model = YOLO('yolov8n.pt')
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

# Globals
tracking_data = {}
scoreboard = defaultdict(int)
prev_people = []
prev_ids = []
tracked_persons = {}
person_id_counter = 1
last_alert_time = defaultdict(lambda: {"hand": None, "head": None})
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
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
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
        tracking_data[pid] = {"left_hand": deque(maxlen=HISTORY_LENGTH), "right_hand": deque(maxlen=HISTORY_LENGTH)}
    tracking_data[pid][joint].append(pt)

def compute_smoothed(pid, joint, pt):
    history = tracking_data[pid][joint]
    if not history:
        return 0
    avg = np.mean(history, axis=0)
    return np.linalg.norm(pt - avg)

def crop_and_save_alert(frame, person, pid, reason):
    global screenshot_count
    valid_points = [p[:2] for p in person if p[2] > 0.1]
    if not valid_points:
        return
    points = np.array(valid_points)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    pad = 30
    x, y = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + 2 * pad, frame.shape[1]), min(y + h + 2 * pad, frame.shape[0])
    cropped = frame[y:y2, x:x2]
    filename = os.path.join(save_dir, f"alert_{pid}_{reason}_{screenshot_count}.jpg")
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

def is_standing(person, threshold_ratio=0.6):
    # Keypoint indices (BODY_25)
    neck = person[1]
    mid_hip = person[8]
    knee_left = person[9]
    ankle_left = person[11]

    # Ensure confidence is good
    if min(neck[2], mid_hip[2], knee_left[2], ankle_left[2]) < 0.3:
        return False

    # Estimate full body height
    height = np.linalg.norm(neck[:2] - ankle_left[:2])
    torso_height = np.linalg.norm(neck[:2] - mid_hip[:2])
    leg_height = np.linalg.norm(mid_hip[:2] - ankle_left[:2])

    # If legs are extended enough → standing
    leg_ratio = leg_height / height

    return leg_ratio > threshold_ratio


def is_getting_down(person, frame_height, shoulder_thresh=0.6, hip_thresh=0.75):
    try:
        left_shoulder_y = person[5][1]
        right_shoulder_y = person[2][1]
        left_hip_y = person[11][1]
        right_hip_y = person[8][1]

        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        avg_hip_y = (left_hip_y + right_hip_y) / 2

        shoulder_down = avg_shoulder_y > frame_height * shoulder_thresh
        hip_down = avg_hip_y > frame_height * hip_thresh

        return shoulder_down and hip_down
    except:
        return False


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
                        "right_shoulder_y": deque(maxlen=HISTORY_LENGTH)
                    }
                else:
                    for key in ["left_shoulder_y", "right_shoulder_y"]:
                        if key not in tracking_data[pid]:
                            tracking_data[pid][key] = deque(maxlen=HISTORY_LENGTH)

                if left_shoulder[2] > 0.4:
                    tracking_data[pid]["left_shoulder_y"].append(left_shoulder[1])
                if right_shoulder[2] > 0.4:
                    tracking_data[pid]["right_shoulder_y"].append(right_shoulder[1])

                if (
                    len(tracking_data[pid]["left_shoulder_y"]) == HISTORY_LENGTH and
                    len(tracking_data[pid]["right_shoulder_y"]) == HISTORY_LENGTH
                ):
                    avg_left = np.mean(tracking_data[pid]["left_shoulder_y"])
                    avg_right = np.mean(tracking_data[pid]["right_shoulder_y"])
                    base_height = max(avg_left, avg_right)  # Conservative standing detection

                    current_left = left_shoulder[1]
                    current_right = right_shoulder[1]
                    current_height = max(current_left, current_right)

                    rise = base_height - current_height  # positive if person moved up

                    if rise > 30:  # ← Adjust this threshold based on test footage
                        if not last_alert_time[pid].get("stand") or now - last_alert_time[pid]["stand"] > COOLDOWN_PERIOD:
                            crop_and_save_alert(frame, person, pid, "standing")
                            scoreboard[pid] += 1
                            last_alert_time[pid]["stand"] = now

                            
                    left_shoulder = person[5]
                    right_shoulder = person[2]

                    # Initialize shoulder Y-history tracking if needed
                    if pid not in tracking_data:
                        tracking_data[pid] = {
                            "left_shoulder_y": deque(maxlen=HISTORY_LENGTH),
                            "right_shoulder_y": deque(maxlen=HISTORY_LENGTH)
                        }
                    else:
                        for key in ["left_shoulder_y", "right_shoulder_y"]:
                            if key not in tracking_data[pid]:
                                tracking_data[pid][key] = deque(maxlen=HISTORY_LENGTH)

                    # Append current shoulder Y positions if confidence is good
                    if left_shoulder[2] > 0.4:
                        tracking_data[pid]["left_shoulder_y"].append(left_shoulder[1])
                    if right_shoulder[2] > 0.4:
                        tracking_data[pid]["right_shoulder_y"].append(right_shoulder[1])

                    # Proceed if enough history is available
                    if (
                        len(tracking_data[pid]["left_shoulder_y"]) == HISTORY_LENGTH and
                        len(tracking_data[pid]["right_shoulder_y"]) == HISTORY_LENGTH
                    ):
                        avg_left = np.mean(tracking_data[pid]["left_shoulder_y"])
                        avg_right = np.mean(tracking_data[pid]["right_shoulder_y"])
                        base_height = max(avg_left, avg_right)  # Baseline: shoulder closer to bottom

                        current_left = left_shoulder[1]
                        current_right = right_shoulder[1]
                        current_height = max(current_left, current_right)

                        drop = current_height - base_height  # Positive if person moved down

                        if drop > 30:  # 🔧 You can tune this threshold
                            if not last_alert_time[pid].get("getdown") or now - last_alert_time[pid]["getdown"] > COOLDOWN_PERIOD:
                                crop_and_save_alert(frame, person, pid, "getdown")
                                scoreboard[pid] += 1
                                last_alert_time[pid]["getdown"] = now






                # === Head turn detection ===
                nose = person[0]
                left_ear, right_ear = person[17], person[18]
                left_eye, right_eye = person[15], person[16]
                if nose[2] <= 0.1:
                    continue

                if pid not in tracked_persons:
                    tracked_persons[pid] = {
                        "nose": tuple(nose[:2]),
                        "baseline_angle": None,
                        "turn_start_time": None,
                        "alerted": False,
                        "excluded": False
                    }
                data = tracked_persons[pid]
                data["nose"] = tuple(nose[:2])
                if data["excluded"]:
                    continue

                # Get facial references
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

                # --- Wall-facing skip logic ---
                x_pos = nose[0]
                frame_width = frame.shape[1]
                edge_margin = 0.15 * frame_width  # 15% edge margin

                if x_pos < edge_margin and rel_angle > 0:
                    continue  # Leftmost person looking right (toward wall) → acceptable

                if x_pos > (frame_width - edge_margin) and rel_angle < 0:
                    continue  # Rightmost person looking left (toward wall) → acceptable

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

    if noise_alert_triggered:
        cv2.putText(output_frame, "🚨 Noise Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        noise_alert_triggered = False

    cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Cheating Detection - Integrated", output_frame)
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