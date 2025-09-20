import os

import cv2
import numpy as np
from collections import deque
import threading
import pyaudio
import audioop

# from config-sp import NOISE_THRESHOLD
# import
from config import Config as cfg

# Listen for noise


class FunSp:
    def listen_for_noise():
        global noise_alert_triggered
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1,
                         rate=16000, input=True, frames_per_buffer=1024)
        while True:
            data = stream.read(1024, exception_on_overflow=False)
            rms = audioop.rms(data, 2)
            if rms > cfg.NOISE_THRESHOLD:
                noise_alert_triggered = True

    threading.Thread(target=listen_for_noise, daemon=True).start()

    # Utils

    def get_body_scale(person):
        if person[5][2] > 0.4 and person[2][2] > 0.4:
            return np.linalg.norm(person[5][:2] - person[2][:2])
        return 1.0

    def update_history(pid, joint, pt):
        if pid not in cfg.tracking_data:
            cfg.tracking_data[pid] = {"left_hand": deque(maxlen=cfg.HISTORY_LENGTH), "right_hand": deque(
                maxlen=cfg.HISTORY_LENGTH), "left_shoulder_y": deque(maxlen=cfg.HISTORY_LENGTH), "right_shoulder_y": deque(maxlen=cfg.HISTORY_LENGTH)}
        cfg.tracking_data[pid][joint].append(pt)

    def compute_smoothed(pid, joint, pt):
        history = cfg.tracking_data[pid][joint]
        if not history:
            return 0
        avg = np.mean(history, axis=0)
        return np.linalg.norm(pt - avg)

    def calculate_distance(p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # def detect_object_passing(person1, person2, pid1, pid2):
    #     # Get hand positions
    #     hands1 = [(person1[4][:2], person1[4][2]),
    #               (person1[7][:2], person1[7][2])]  # Right, Left
    #     hands2 = [(person2[4][:2], person2[4][2]),
    #               (person2[7][:2], person2[7][2])]

    #     min_distance = float('inf')
    #     passing_detected = False

    #     for hand1_pos, hand1_conf in hands1:
    #         if hand1_conf < 0.3:
    #             continue
    #         for hand2_pos, hand2_conf in hands2:
    #             if hand2_conf < 0.3:
    #                 continue

    #             dist = calculate_distance(hand1_pos, hand2_pos)
    #             if dist < min_distance:
    #                 min_distance = dist

    #             if dist < cfg.OBJECT_PASSING_THRESHOLD:
    #                 passing_detected = True

    #     return passing_detected, min_distance

    def crop_and_save_alert(frame, person, pid, reason):
        # global cfg.screenshot_count 

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
            cfg.save_dir, f"alert_{pid}_{reason}_{cfg.screenshot_count}.jpg")
        cv2.imwrite(filename, cropped)
        print(f"⚠️ {pid} {reason} alert -> {filename}")
        cfg.screenshot_count += 1

    def assign_ids(current, previous, previous_ids, threshold=1000):
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
                new_ids[i] = f"ID_{cfg.person_id_counter}"
                cfg.person_id_counter += 1
        return new_ids
