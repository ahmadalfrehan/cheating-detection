import pyopenpose as op
import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime, timedelta
# from openpose_setup import OpenposeSetup as ops
from config import Config as cfg
from funsp import FunSp as fs
# Main Loop
import os
from functions import handmovement, detectstanding, horizantalmovement
import sys
os.add_dll_directory(r"C:/openpose/build/x64/Release")
os.add_dll_directory(
    r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
sys.path.append(r"C:/openpose/build/python/openpose/Release")
params = {
    "model_folder": "C:/openpose/models",
    "model_pose": "BODY_25",
    "disable_blending": False,
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Video & YOLO
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cfg.frame_count += 1
    elapsed_time = time.time() - start_time
    fps = cfg.frame_count / elapsed_time if elapsed_time > 0 else 0
    output_frame = frame.copy()
    current_time = time.time()

    if current_time - cfg.last_process_time >= cfg.PROCESS_EVERY_SECONDS:
        cfg.last_process_time = current_time

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
            ids = fs.assign_ids(keypoints, cfg.prev_people, cfg.prev_ids)
            cfg.prev_people, cfg.prev_ids = keypoints.copy(), ids.copy()

            # # === Object Passing Detection (pairwise) ===
            # for i in range(len(keypoints)):
            #     for j in range(i + 1, len(keypoints)):
            #         pid1, pid2 = ids[i], ids[j]
            #         person1, person2 = keypoints[i], keypoints[j]

            #         if person1 is None or person2 is None:
            #             continue

            #         passing, min_dist = fs.detect_object_passing(
            #             person1, person2, pid1, pid2)
            #         if passing:
            #             fs.crop_and_save_alert(
            #                 frame, person1, pid1, f"passing_{pid2}")
            #             fs.crop_and_save_alert(
            #                 frame, person2, pid2, f"passing_{pid1}")
            #             cfg.scoreboard[pid1] += 1
            #             cfg.scoreboard[pid2] += 1
            #             print(
            #                 f"🔁 Object passing detected between {pid1} and {pid2} (distance: {min_dist:.1f})")

            for i, person in enumerate(keypoints):

                pid = ids[i]
                scale = fs.get_body_scale(person)
                hand_centers = []
                now = datetime.now()

                # === Hand movement ===
                if pid in cfg.tracked_persons and cfg.tracked_persons[pid]["excluded"]:
                    continue  # Skip hand detection for excluded people

                hand_centers = handmovement(pid, person, frame, now, scale)

                # === Shoulder movement (standing detection) ===
                left_shoulder = person[5]
                right_shoulder = person[2]

                nose = detectstanding(pid, person, frame, now)
                # === Horizontal Movement Detection ===
                horizantalmovement(pid, person, left_shoulder,
                                   right_shoulder, frame, now, nose)
                # === Head turn detection ===
                left_ear, right_ear = person[17], person[18]
                left_eye, right_eye = person[15], person[16]
                if nose[2] <= 0.1:
                    continue

                if pid not in cfg.tracked_persons:
                    cfg.tracked_persons[pid] = {
                        "disappear_start_time": None,
                        "disappeared": False,
                        "nose": tuple(nose[:2]),
                        "baseline_angle": None,
                        "turn_start_time": None,
                        "alerted": False,
                        "excluded": False
                    }
                elif "disappear_start_time" not in cfg.tracked_persons[pid]:
                    cfg.tracked_persons[pid]["disappear_start_time"] = None
                    cfg.tracked_persons[pid]["disappeared"] = False
                shoulder_visible = left_shoulder[2] > 0.1 or right_shoulder[2] > 0.1
                data = cfg.tracked_persons[pid]
                if not shoulder_visible:
                    if data["disappear_start_time"] is None:
                        data["disappear_start_time"] = time.time()
                    elif not data["disappeared"] and time.time() - data["disappear_start_time"] > 3:
                        fs.crop_and_save_alert(frame, person, pid, "disappear")
                        cfg.scoreboard[pid] += 1
                        data["disappeared"] = True
                else:
                    # Reset if shoulders reappear
                    data["disappear_start_time"] = None
                    data["disappeared"] = False
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
                    if abs(angle) > cfg.MAX_BASELINE_ALLOWED:
                        data["excluded"] = True
                        continue
                    data["baseline_angle"] = angle

                rel_angle = angle - data["baseline_angle"]

                # --- Wall-facing skip logic ---
                x_pos = nose[0]
                frame_width = frame.shape[1]
                edge_margin = 0.1 * frame_width  # 15% edge margin

                if x_pos < edge_margin and rel_angle > 0:
                    # Leftmost person looking right (toward wall) → acceptable
                    continue

                if x_pos > (frame_width - edge_margin) and rel_angle < 0:
                    # Rightmost person looking left (toward wall) → acceptable
                    continue

                # Symmetry check
                symmetry_ratio = None
                try:
                    symmetry_ratio = abs(
                        left_ear[0] - nose[0]) / abs(right_ear[0] - nose[0])
                except:
                    pass
                sym_ok = symmetry_ratio is None or symmetry_ratio < cfg.SYM_RATIO_MIN or symmetry_ratio > cfg.SYM_RATIO_MAX

                head_turn = (
                    abs(rel_angle) > cfg.ANGLE_THRESHOLD and
                    abs(rel_angle) > cfg.MIN_DEVIATION_FROM_BASELINE and
                    sym_ok
                )

                # Trigger head movement alert
                if head_turn:
                    if data["turn_start_time"] is None:
                        data["turn_start_time"] = current_time
                    elif current_time - data["turn_start_time"] >= cfg.MIN_HOLD_TIME and not data["alerted"]:
                        if not cfg.last_alert_time[pid]["head"] or now - cfg.last_alert_time[pid]["head"] >= cfg.COOLDOWN_PERIOD:
                            cfg.scoreboard[pid] += 1
                            fs.crop_and_save_alert(frame, person, pid, "head")
                            cfg.last_alert_time[pid]["head"] = now
                            data["alerted"] = True
                else:
                    data["turn_start_time"] = None
                    data["alerted"] = False
        else:
            # If no one is detected in the frame, mark all currently tracked people as possibly disappeared
            now = time.time()
            for pid, data in cfg.tracked_persons.items():
                if data["disappear_start_time"] is None:
                    data["disappear_start_time"] = now
                elif not data["disappeared"] and now - data["disappear_start_time"] > 3:
                    print(f"🚨 {pid} disappeared (no keypoints at all)")
                    # Use full frame since no person crop
                    fs.crop_and_save_alert(frame, frame, pid, "disappear")
                    cfg.scoreboard[pid] += 1
                    data["disappeared"] = True

    if cfg.noise_alert_triggered:
        cv2.putText(output_frame, "🚨 Noise Detected!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        noise_alert_triggered = False

    cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Cheating Detection - Integrated", output_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if cfg.scoreboard:
    print("\n📊 FINAL CHEATING REPORT:")
    for student, score in cfg.scoreboard.items():
        print(f"🧑 {student}: {score} cheating points")
else:
    print("\n📊 No cheating detected during the session.")
