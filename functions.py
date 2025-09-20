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
import sys


def handmovement(pid, person, frame, now, scale):
    hand_centers = []
    for joint_name, joint in {"left_hand": person[7], "right_hand": person[4]}.items():
        if joint[2] < 0.4:
            continue
        fs.update_history(pid, joint_name, joint[:2])
        norm = fs.compute_smoothed(pid, joint_name, joint[:2]) / scale
        hand_centers.append(joint[:2])

        if cfg.last_alert_time[pid]["hand"] and now - cfg.last_alert_time[pid]["hand"] < cfg.COOLDOWN_PERIOD:
            continue

        if norm >= cfg.THRESHOLD:
            fs.crop_and_save_alert(frame, person, pid, "hand")
            cfg.scoreboard[pid] += 1
            cfg.last_alert_time[pid]["hand"] = now
    return hand_centers


def detectstanding(pid, person, frame, now):
    left_shoulder = person[5]
    right_shoulder = person[2]

    # Ensure shoulder and hand history keys are initialized
    if pid not in cfg.tracking_data:
        cfg.tracking_data[pid] = {
            "left_hand": deque(maxlen=cfg.HISTORY_LENGTH),
            "right_hand": deque(maxlen=cfg.HISTORY_LENGTH),
            "left_shoulder_y": deque(maxlen=cfg.HISTORY_LENGTH),
            "right_shoulder_y": deque(maxlen=cfg.HISTORY_LENGTH)
        }
    else:
        for key in ["left_shoulder_y", "right_shoulder_y"]:
            if key not in cfg.tracking_data[pid]:
                cfg.tracking_data[pid][key] = deque(maxlen=cfg.HISTORY_LENGTH)

    if left_shoulder[2] > 0.4:
        cfg.tracking_data[pid]["left_shoulder_y"].append(left_shoulder[1])
    if right_shoulder[2] > 0.4:
        cfg.tracking_data[pid]["right_shoulder_y"].append(right_shoulder[1])

    if (len(cfg.tracking_data[pid]["left_shoulder_y"]) == cfg.HISTORY_LENGTH and
            len(cfg.tracking_data[pid]
                ["right_shoulder_y"]) == cfg.HISTORY_LENGTH
            ):
        avg_left = np.mean(cfg.tracking_data[pid]["left_shoulder_y"])
        avg_right = np.mean(cfg.tracking_data[pid]["right_shoulder_y"])
        # Conservative standing detection
        base_height = max(avg_left, avg_right)

        current_left = left_shoulder[1]
        current_right = right_shoulder[1]
        current_height = max(current_left, current_right)

        rise = base_height - current_height  # positive if person moved up

        if rise > 30:  # ← Adjust this cfg.threshold based on test footage
            if not cfg.last_alert_time[pid].get("stand") or now - cfg.last_alert_time[pid]["stand"] > cfg.COOLDOWN_PERIOD:
                fs.crop_and_save_alert(frame, person, pid, "standing")
                cfg.scoreboard[pid] += 1
                cfg.last_alert_time[pid]["stand"] = now

    nose = person[0]
    return nose


def horizantalmovement(pid, person, left_shoulder, right_shoulder, frame, now, nose):
    if pid not in cfg.tracked_persons:
        cfg.tracked_persons[pid] = {
            "disappear_start_time": None,
            "disappeared": False,
            "nose": tuple(nose[:2]),
            "baseline_angle": None,
            "turn_start_time": None,
            "alerted": False,
            "excluded": False,
            "baseline_x": None,
            "horizontal_alerted": False
        }

    if left_shoulder[2] > 0.4 and right_shoulder[2] > 0.4:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        if cfg.tracked_persons[pid]["baseline_x"] is None:
            cfg.tracked_persons[pid]["baseline_x"] = shoulder_center_x
        else:
            delta_x = abs(shoulder_center_x -
                          cfg.tracked_persons[pid]["baseline_x"])
            if "last_horizontal_alert" not in cfg.tracked_persons[pid]:
                cfg.tracked_persons[pid]["last_horizontal_alert"] = None

                # Check movement and cooldown
            if delta_x > 50:
                last_alert = cfg.tracked_persons[pid]["last_horizontal_alert"]
                if not last_alert or now - last_alert > timedelta(seconds=5):
                    fs.crop_and_save_alert(
                        frame, person, pid, "horizontal_movement")
                    cfg.scoreboard[pid] += 1
                    cfg.tracked_persons[pid]["last_horizontal_alert"] = now


# def headturndetection():
