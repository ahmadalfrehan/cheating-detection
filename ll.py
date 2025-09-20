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
import math

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

# Video setup
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Existing Constants
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
DOWN_THRESHOLD = 40
DOWN_HOLD_TIME = 0.3

# NEW: Advanced detection constants
POSITION_CHANGE_THRESHOLD = 80  # Pixels moved to consider position change
MIN_POSITION_CHANGE_TIME = 2.0  # Minimum time to trigger position change alert
PROXIMITY_THRESHOLD = 120  # Minimum distance between people (pixels)
PROXIMITY_ALERT_TIME = 3.0  # Time people must be close to trigger alert
COORDINATION_THRESHOLD = 0.8  # Similarity threshold for coordinated movements
COORDINATION_WINDOW = 5  # Frames to analyze for coordination
POINTING_ANGLE_THRESHOLD = 45  # Degrees for pointing detection
POINTING_DISTANCE_THRESHOLD = 200  # Max distance for pointing target
OBJECT_PASSING_THRESHOLD = 50  # Distance for object passing detection
OBJECT_PASSING_TIME = 1.0  # Time window for object passing

# Globals
tracking_data = {}
scoreboard = defaultdict(int)
prev_people = []
prev_ids = []
tracked_persons = {}
person_id_counter = 1
last_alert_time = defaultdict(lambda: {
    "hand": None, "head": None, "stand": None, "down": None,
    "position": None, "proximity": None, "coordination": None, 
    "pointing": None, "object_passing": None
})
screenshot_count = 0
frame_count = 0
last_process_time = 0
noise_alert_triggered = False
save_dir = "alerts"
os.makedirs(save_dir, exist_ok=True)

# NEW: Advanced tracking data
coordination_history = deque(maxlen=COORDINATION_WINDOW)
proximity_tracker = {}
position_baselines = {}

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

def get_body_scale(person):
    if person[5][2] > 0.4 and person[2][2] > 0.4:
        return np.linalg.norm(person[5][:2] - person[2][:2])
    return 1.0

def get_person_center(person):
    """Get center point of person for position tracking"""
    valid_points = [p[:2] for p in person if p[2] > 0.3]
    if not valid_points:
        return None
    return np.mean(valid_points, axis=0)

def get_torso_center(person):
    """Get torso center for more stable position tracking"""
    # Use neck, shoulders for torso center
    torso_points = []
    if person[1][2] > 0.3:  # Neck
        torso_points.append(person[1][:2])
    if person[2][2] > 0.3:  # Right shoulder
        torso_points.append(person[2][:2])
    if person[5][2] > 0.3:  # Left shoulder
        torso_points.append(person[5][:2])
    
    if not torso_points:
        return None
    return np.mean(torso_points, axis=0)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    if p1 is None or p2 is None:
        return float('inf')
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect_pointing_gesture(person, all_people, person_id):
    """Detect if person is pointing at another person"""
    # Get arm keypoints
    right_shoulder = person[2]  # Right shoulder
    right_elbow = person[3]     # Right elbow  
    right_wrist = person[4]     # Right wrist
    
    left_shoulder = person[5]   # Left shoulder
    left_elbow = person[6]      # Left elbow
    left_wrist = person[7]      # Left wrist
    
    pointing_targets = []
    
    # Check right arm pointing
    if (right_shoulder[2] > 0.3 and right_elbow[2] > 0.3 and right_wrist[2] > 0.3):
        # Calculate arm angle
        shoulder_to_elbow = right_elbow[:2] - right_shoulder[:2]
        elbow_to_wrist = right_wrist[:2] - right_elbow[:2]
        
        # Check if arm is extended (pointing gesture)
        arm_angle = np.arccos(np.dot(shoulder_to_elbow, elbow_to_wrist) / 
                             (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist)))
        arm_angle = np.degrees(arm_angle)
        
        if arm_angle < POINTING_ANGLE_THRESHOLD:  # Arm is relatively straight
            # Check if pointing at someone
            wrist_pos = right_wrist[:2]
            for i, other_person in enumerate(all_people):
                if i == person_id:  # Don't point at self
                    continue
                other_center = get_person_center(other_person)
                if other_center is not None:
                    dist = calculate_distance(wrist_pos, other_center)
                    if dist < POINTING_DISTANCE_THRESHOLD:
                        pointing_targets.append(f"ID_{i}")
    
    # Check left arm pointing
    if (left_shoulder[2] > 0.3 and left_elbow[2] > 0.3 and left_wrist[2] > 0.3):
        shoulder_to_elbow = left_elbow[:2] - left_shoulder[:2]
        elbow_to_wrist = left_wrist[:2] - left_elbow[:2]
        
        arm_angle = np.arccos(np.dot(shoulder_to_elbow, elbow_to_wrist) / 
                             (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist)))
        arm_angle = np.degrees(arm_angle)
        
        if arm_angle < POINTING_ANGLE_THRESHOLD:
            wrist_pos = left_wrist[:2]
            for i, other_person in enumerate(all_people):
                if i == person_id:
                    continue
                other_center = get_person_center(other_person)
                if other_center is not None:
                    dist = calculate_distance(wrist_pos, other_center)
                    if dist < POINTING_DISTANCE_THRESHOLD:
                        pointing_targets.append(f"ID_{i}")
    
    return pointing_targets

def detect_object_passing(person1, person2, pid1, pid2):
    """Detect if objects are being passed between people"""
    # Get hand positions
    hands1 = [(person1[4][:2], person1[4][2]), (person1[7][:2], person1[7][2])]  # Right, Left
    hands2 = [(person2[4][:2], person2[4][2]), (person2[7][:2], person2[7][2])]
    
    min_distance = float('inf')
    passing_detected = False
    
    for hand1_pos, hand1_conf in hands1:
        if hand1_conf < 0.3:
            continue
        for hand2_pos, hand2_conf in hands2:
            if hand2_conf < 0.3:
                continue
            
            dist = calculate_distance(hand1_pos, hand2_pos)
            if dist < min_distance:
                min_distance = dist
            
            if dist < OBJECT_PASSING_THRESHOLD:
                passing_detected = True
    
    return passing_detected, min_distance

def calculate_movement_similarity(movements1, movements2):
    """Calculate similarity between two movement patterns"""
    if len(movements1) != len(movements2) or len(movements1) == 0:
        return 0.0
    
    # Calculate correlation between movement vectors
    mov1 = np.array(movements1)
    mov2 = np.array(movements2)
    
    if np.std(mov1) == 0 or np.std(mov2) == 0:
        return 0.0
    
    correlation = np.corrcoef(mov1.flatten(), mov2.flatten())[0, 1]
    return abs(correlation) if not np.isnan(correlation) else 0.0

def update_history(pid, joint, pt):
    if pid not in tracking_data:
        tracking_data[pid] = {
            "left_hand": deque(maxlen=HISTORY_LENGTH),
            "right_hand": deque(maxlen=HISTORY_LENGTH),
            "left_shoulder_y": deque(maxlen=HISTORY_LENGTH),
            "right_shoulder_y": deque(maxlen=HISTORY_LENGTH),
            "head_y": deque(maxlen=HISTORY_LENGTH),
            "torso_y": deque(maxlen=HISTORY_LENGTH),
            "center_position": deque(maxlen=HISTORY_LENGTH),  # NEW
            "movement_vector": deque(maxlen=COORDINATION_WINDOW)  # NEW
        }
    
    # Ensure all keys exist
    required_keys = ["center_position", "movement_vector"]
    for key in required_keys:
        if key not in tracking_data[pid]:
            if key == "movement_vector":
                tracking_data[pid][key] = deque(maxlen=COORDINATION_WINDOW)
            else:
                tracking_data[pid][key] = deque(maxlen=HISTORY_LENGTH)
    
    tracking_data[pid][joint].append(pt)

def compute_smoothed(pid, joint, pt):
    history = tracking_data[pid][joint]
    if not history:
        return 0
    avg = np.mean(history, axis=0)
    return np.linalg.norm(pt - avg)

def crop_and_save_alert(frame, person, pid, reason):
    global screenshot_count
    
    if isinstance(person, np.ndarray) and person.ndim == 2 and person.shape[1] == 3:
        valid_points = [p[:2] for p in person if p[2] > 0.1]
        if not valid_points:
            cropped = frame.copy()
        else:
            points = np.array(valid_points)
            x, y, w, h = cv2.boundingRect(points.astype(np.int32))
            pad = 30
            x, y = max(x - pad, 0), max(y - pad, 0)
            x2, y2 = min(x + w + 2 * pad, frame.shape[1]), min(y + h + 2 * pad, frame.shape[0])
            cropped = frame[y:y2, x:x2]
    else:
        cropped = frame.copy()

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
            keypoints = np.array(filtered_keypoints) if filtered_keypoints else None
        else:
            keypoints = None

        if keypoints is not None and keypoints.shape[0] > 0:
            ids = assign_ids(keypoints, prev_people, prev_ids)
            prev_people, prev_ids = keypoints.copy(), ids.copy()

            # Collect current frame data for coordination analysis
            current_movements = {}
            current_positions = {}
            
            for i, person in enumerate(keypoints):
                pid = ids[i]
                scale = get_body_scale(person)
                now = datetime.now()
                
                # === Position Change Detection ===
                current_center = get_torso_center(person)
                if current_center is not None:
                    current_positions[pid] = current_center
                    
                    # Initialize position baseline
                    if pid not in position_baselines:
                        position_baselines[pid] = {
                            "baseline": current_center.copy(),
                            "last_position": current_center.copy(),
                            "position_change_start": None
                        }
                    
                    # Check for significant position change
                    baseline_dist = calculate_distance(current_center, position_baselines[pid]["baseline"])
                    
                    if baseline_dist > POSITION_CHANGE_THRESHOLD:
                        if position_baselines[pid]["position_change_start"] is None:
                            position_baselines[pid]["position_change_start"] = current_time
                        elif (current_time - position_baselines[pid]["position_change_start"] >= MIN_POSITION_CHANGE_TIME):
                            if (not last_alert_time[pid]["position"] or 
                                now - last_alert_time[pid]["position"] >= COOLDOWN_PERIOD):
                                crop_and_save_alert(frame, person, pid, "position_change")
                                scoreboard[pid] += 1
                                last_alert_time[pid]["position"] = now
                                print(f"📍 {pid} position change detected (moved {baseline_dist:.1f}px)")
                                # Update baseline to new position
                                position_baselines[pid]["baseline"] = current_center.copy()
                                position_baselines[pid]["position_change_start"] = None
                    else:
                        position_baselines[pid]["position_change_start"] = None
                    
                    # Track movement for coordination analysis
                    if pid in tracking_data and "center_position" in tracking_data[pid]:
                        if len(tracking_data[pid]["center_position"]) > 0:
                            last_pos = tracking_data[pid]["center_position"][-1]
                            movement = current_center - last_pos
                            current_movements[pid] = movement
                            
                            # Update movement history
                            update_history(pid, "movement_vector", movement)
                    
                    update_history(pid, "center_position", current_center)

                # === Pointing Gesture Detection ===
                pointing_targets = detect_pointing_gesture(person, keypoints, i)
                if pointing_targets:
                    if (not last_alert_time[pid]["pointing"] or 
                        now - last_alert_time[pid]["pointing"] >= COOLDOWN_PERIOD):
                        crop_and_save_alert(frame, person, pid, "pointing")
                        scoreboard[pid] += 1
                        last_alert_time[pid]["pointing"] = now
                        print(f"👉 {pid} pointing at {', '.join(pointing_targets)}")

                # === Existing Detection Logic ===
                if pid in tracked_persons and tracked_persons[pid].get("excluded", False):
                    continue

                # Hand movement detection (existing code)
                for joint_name, joint in {"left_hand": person[7], "right_hand": person[4]}.items():
                    if joint[2] < 0.4:
                        continue
                    update_history(pid, joint_name, joint[:2])
                    norm = compute_smoothed(pid, joint_name, joint[:2]) / scale

                    if last_alert_time[pid]["hand"] and now - last_alert_time[pid]["hand"] < COOLDOWN_PERIOD:
                        continue

                    if norm >= THRESHOLD:
                        crop_and_save_alert(frame, person, pid, "hand")
                        scoreboard[pid] += 1
                        last_alert_time[pid]["hand"] = now

                # [Rest of existing detection code - standing, getting down, head turn, etc.]
                # ... (keeping existing code for brevity)

            # === Proximity Detection ===
            if len(current_positions) > 1:
                people_list = list(current_positions.keys())
                for i in range(len(people_list)):
                    for j in range(i + 1, len(people_list)):
                        pid1, pid2 = people_list[i], people_list[j]
                        pos1, pos2 = current_positions[pid1], current_positions[pid2]
                        
                        distance = calculate_distance(pos1, pos2)
                        pair_key = tuple(sorted([pid1, pid2]))
                        
                        if distance < PROXIMITY_THRESHOLD:
                            if pair_key not in proximity_tracker:
                                proximity_tracker[pair_key] = current_time
                            elif current_time - proximity_tracker[pair_key] >= PROXIMITY_ALERT_TIME:
                                # Check if either person hasn't been alerted recently
                                can_alert = True
                                for pid in [pid1, pid2]:
                                    if (last_alert_time[pid]["proximity"] and 
                                        datetime.now() - last_alert_time[pid]["proximity"] < COOLDOWN_PERIOD):
                                        can_alert = False
                                        break
                                
                                if can_alert:
                                    # Alert both people
                                    for pid in [pid1, pid2]:
                                        person_idx = people_list.index(pid)
                                        crop_and_save_alert(frame, keypoints[person_idx], pid, "proximity")
                                        scoreboard[pid] += 1
                                        last_alert_time[pid]["proximity"] = datetime.now()
                                    print(f"🤝 {pid1} and {pid2} too close (distance: {distance:.1f}px)")
                                
                                proximity_tracker[pair_key] = current_time
                        else:
                            if pair_key in proximity_tracker:
                                del proximity_tracker[pair_key]

                        # === Object Passing Detection ===
                        person1 = keypoints[people_list.index(pid1)]
                        person2 = keypoints[people_list.index(pid2)]
                        passing_detected, hand_distance = detect_object_passing(person1, person2, pid1, pid2)
                        
                        if passing_detected:
                            can_alert = True
                            for pid in [pid1, pid2]:
                                if (last_alert_time[pid]["object_passing"] and 
                                    datetime.now() - last_alert_time[pid]["object_passing"] < COOLDOWN_PERIOD):
                                    can_alert = False
                                    break
                            
                            if can_alert:
                                for pid in [pid1, pid2]:
                                    person_idx = people_list.index(pid)
                                    crop_and_save_alert(frame, keypoints[person_idx], pid, "object_passing")
                                    scoreboard[pid] += 1
                                    last_alert_time[pid]["object_passing"] = datetime.now()
                                print(f"📦 {pid1} and {pid2} object passing detected (hand distance: {hand_distance:.1f}px)")

            # === Coordinated Movement Detection ===
            if len(current_movements) > 1:
                coordination_history.append(current_movements)
                
                if len(coordination_history) == COORDINATION_WINDOW:
                    # Analyze coordination over the window
                    people_list = list(current_movements.keys())
                    for i in range(len(people_list)):
                        for j in range(i + 1, len(people_list)):
                            pid1, pid2 = people_list[i], people_list[j]
                            
                            # Extract movement sequences for both people
                            movements1 = []
                            movements2 = []
                            
                            for frame_movements in coordination_history:
                                if pid1 in frame_movements and pid2 in frame_movements:
                                    movements1.append(frame_movements[pid1])
                                    movements2.append(frame_movements[pid2])
                            
                            if len(movements1) >= 3:  # Need sufficient data
                                similarity = calculate_movement_similarity(movements1, movements2)
                                
                                if similarity > COORDINATION_THRESHOLD:
                                    pair_key = tuple(sorted([pid1, pid2]))
                                    can_alert = True
                                    for pid in [pid1, pid2]:
                                        if (last_alert_time[pid]["coordination"] and 
                                            datetime.now() - last_alert_time[pid]["coordination"] < COOLDOWN_PERIOD):
                                            can_alert = False
                                            break
                                    
                                    if can_alert:
                                        for pid in [pid1, pid2]:
                                            person_idx = people_list.index(pid)
                                            crop_and_save_alert(frame, keypoints[person_idx], pid, "coordination")
                                            scoreboard[pid] += 1
                                            last_alert_time[pid]["coordination"] = datetime.now()
                                        print(f"🤖 {pid1} and {pid2} coordinated movement detected (similarity: {similarity:.2f})")

    # Display alerts and info
    if noise_alert_triggered:
        cv2.putText(output_frame, "🚨 Noise Detected!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        noise_alert_triggered = False

    cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Show active proximities
    y_offset = 100
    for pair, start_time in proximity_tracker.items():
        time_close = current_time - start_time
        cv2.putText(output_frame, f"{pair[0]} & {pair[1]} close: {time_close:.1f}s", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        y_offset += 25

    cv2.imshow("Advanced Cheating Detection", output_frame)
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