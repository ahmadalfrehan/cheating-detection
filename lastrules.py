from flask_socketio import SocketIO, emit
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from man import broadcast_alert, update_video_frame
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
import winsound
import face_recognition
import speech_recognition as sr
from threading import Thread, Lock
from flask import Flask, Response, send_from_directory


os.add_dll_directory(r"C:/openpose/build/x64/Release")
os.add_dll_directory(
    r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
sys.path.append(r"C:/openpose/build/python/openpose/Release")


params = {
    "model_folder": "C:/openpose/models",
    "model_pose": "BODY_25",
    "net_resolution": "-1x160",
    "disable_blending": False,
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


THRESHOLD = 1.1
COOLDOWN_PERIOD = timedelta(seconds=2)
NOISE_THRESHOLD = 100
HISTORY_LENGTH = 10
PROCESS_EVERY_SECONDS = 0.3

ANGLE_THRESHOLD = 20
MAX_BASELINE_ALLOWED = 100
MIN_DEVIATION_FROM_BASELINE = 6
MIN_HOLD_TIME = 0.5
SYM_RATIO_MIN = 0.7
SYM_RATIO_MAX = 1.3


MIN_TOTAL_SCORE = 8

RULES = [
    {"min_score": 1, "max_score": 3, "action": "⚠️ Warning"},
    {"min_score": 4, "max_score": 6, "action": "❗ Under Review"},
    {"min_score": 7, "max_score": float("inf"), "action": "⛔ Disqualified"},
]


tracking_data = {}

id_freeze_until = 5
frozen_ids = []
frozen_positions = []  # list of [x, y] shoulder centers

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

face_db = {}
face_id_map = {}

cheating_keywords = [
    "answer", "answers", "copy", "cheat", "cheating", "help", "show me", "what’s the answer",
    "google", "whisper", "pass", "send me", "solution", "search", "read it", "check online",
    "take a picture", "solve this", "look up", "i don’t know", "repeat the question",
    "a", "b", "c", "d", "ay", "bee", "cee", "dee",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",

    "حل", "الاجابة", "الجواب", "ساعدني", "انسخ", "غش", "اغش", "قول", "صديقي", "قوللي",
    "عيد", "كرري", "الصورة", "صور", "ابحث", "النت", "جوجل", "قوقل", "ارسل", "جاوبني",
    "ساعد", "هات", "اجب", "شوف", "ارني", "الكتاب", "حل السؤال", "هات الحل",
    "ما اعرف", "مش عارف", "دور", "ممكن تساعدني",
    "واحد", "اثنين", "ثلاثة", "اربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"
]


latest_frame = None
frame_lock = Lock()


def recognize_bilingual(audio):
    try:
        # Try Arabic
        text = recognizer.recognize_google(audio, language="ar-SA")
        print(f"🗣️ Arabic: {text}")
        return text
    except sr.UnknownValueError:
        pass

    try:
        # Try English
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"🗣️ English: {text}")
        return text
    except sr.UnknownValueError:
        return None


recognizer = sr.Recognizer()
mic = sr.Microphone()


def listen_and_transcribe():
    global noise_alert_triggered
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎤 Speech recognition ready.")
        while True:
            try:
                audio = recognizer.listen(
                    source, timeout=5, phrase_time_limit=5)
                text = recognize_bilingual(audio)

                if text:
                    lowered = text.lower()
                    print(f"📝 Transcription: {text}")
                    with open("transcriptions.txt", "a", encoding="utf-8") as f:
                        f.write(f"{datetime.now()} - {text}\n")

                    # Check for cheating keywords
                    for word in cheating_keywords:
                        if word in lowered:
                            print("🚨 Suspicious speech detected!")
                            noise_alert_triggered = True  # or take screenshot
                            break
            except sr.WaitTimeoutError:
                continue


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

# threading.Thread(target=listen_for_noise, daemon=True).start()
# threading.Thread(target=listen_and_transcribe, daemon=True).start()


def play_beep_loop(pid):
    while tracked_persons.get(pid, {}).get("decision") == "⛔ Disqualified" and not tracked_persons[pid].get("disappeared", False):
        winsound.Beep(1000, 500)
        time.sleep(0.5)


def get_body_scale(person):
    if person[5][2] > 0.4 and person[2][2] > 0.4:
        return np.linalg.norm(person[5][:2] - person[2][:2])
    return 1.0


def update_history(pid, joint, pt):
    if pid not in tracking_data:
        tracking_data[pid] = {"left_hand": deque(maxlen=HISTORY_LENGTH), "right_hand": deque(
            maxlen=HISTORY_LENGTH), "left_shoulder_y": deque(maxlen=HISTORY_LENGTH), "right_shoulder_y": deque(maxlen=HISTORY_LENGTH)}
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
            x2, y2 = min(x + w + 2 * pad,
                         frame.shape[1]), min(y + h + 2 * pad, frame.shape[0])
            cropped = frame[y:y2, x:x2]
    else:
        cropped = frame.copy()

    filename = os.path.join(
        save_dir, f"alert_{pid}_{reason}_{screenshot_count}.jpg")
    cv2.imwrite(filename, cropped)
    print(f"⚠️ {pid} {reason} alert -> {filename}")
    screenshot_count += 1
    return filename


# def assign_ids(current, previous, previous_ids, threshold=100):
#     global person_id_counter
#     new_ids = [None] * len(current)
#     used = set()

#     for i, p in enumerate(current):
#         best_score = float("inf")
#         best_match = -1
#         for j, q in enumerate(previous):
#             if j in used:
#                 continue

#             confident_points = [k for k in [1, 2, 5, 0] if p[k][2] > 0.3 and q[k][2] > 0.3]
#             if len(confident_points) < 2:
#                 continue

#             # Distance between necks, shoulders, and nose
#             neck_score = np.linalg.norm(p[1][:2] - q[1][:2]) if p[1][2] > 0.2 and q[1][2] > 0.2 else 0
#             l_shoulder_score = np.linalg.norm(p[5][:2] - q[5][:2]) if p[5][2] > 0.2 and q[5][2] > 0.2 else 0
#             r_shoulder_score = np.linalg.norm(p[2][:2] - q[2][:2]) if p[2][2] > 0.2 and q[2][2] > 0.2 else 0
#             nose_score = np.linalg.norm(p[0][:2] - q[0][:2]) if p[0][2] > 0.2 and q[0][2] > 0.2 else 0

#             # Weighted sum
#             total_score = (
#                 0.4 * neck_score +
#                 0.2 * l_shoulder_score +
#                 0.2 * r_shoulder_score +
#                 0.2 * nose_score
#             )

#             if total_score < best_score:
#                 best_score = total_score
#                 best_match = j

#         if best_match != -1 and best_score < threshold and best_score > 0 and best_score < MIN_TOTAL_SCORE:
#             new_ids[i] = previous_ids[best_match]
#             used.add(best_match)
#         else:
#             new_ids[i] = f"ID_{person_id_counter}"
#             person_id_counter += 1

#     return new_ids

def assign_ids_with_faces(current_keypoints, frame, threshold=0.6):
    global person_id_counter, face_db

    new_ids = [None] * len(current_keypoints)
    used_ids = set()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for i, encoding in enumerate(face_encodings):
        # for i, (keypoints, encoding) in enumerate(zip(current_keypoints, face_encodings)):
        matched_id = None
        best_match_id = None
        smallest_distance = threshold

        for known_id, known_encodings in face_db.items():
            distances = face_recognition.face_distance(
                known_encodings, encoding)
            min_distance = min(distances, default=1.0)
            if min_distance < smallest_distance:
                best_match_id = known_id
                smallest_distance = min_distance

        if best_match_id:
            matched_id = best_match_id
        if matched_id:
            new_ids[i] = matched_id
            used_ids.add(matched_id)
        else:
            new_id = f"ID_{person_id_counter}"
            person_id_counter += 1
            new_ids[i] = new_id
            face_db[new_id] = [encoding]
            # face_db[matched_id].append(encoding)
            used_ids.add(new_id)

    return new_ids


missing_ids = deque()  # Add this at the top near `person_id_counter`


def assign_ids(current, previous, previous_ids, threshold=1000):
    global person_id_counter, missing_ids

    new_ids = [None] * len(current)
    used = set()
    current_active_ids = set()

    for i, p in enumerate(current):
        best_dist = float("inf")
        match = -1
        for j, q in enumerate(previous):
            if j in used:
                continue
            if p[1][2] < 0.2 or q[1][2] < 0.2:
                continue
            dist = np.linalg.norm(p[1][:2] - q[1][:2])
            if dist < best_dist:
                best_dist = dist
                match = j
        if match != -1 and best_dist < threshold:
            new_ids[i] = previous_ids[match]
            used.add(match)
            current_active_ids.add(new_ids[i])
        else:
            if missing_ids:
                reused_id = missing_ids.popleft()
                new_ids[i] = reused_id
            else:
                new_ids[i] = f"ID_{person_id_counter}"
                person_id_counter += 1
            current_active_ids.add(new_ids[i])

    # Update active and missing IDs
    previous_id_set = set(previous_ids)
    disappeared = previous_id_set - current_active_ids
    for pid in disappeared:
        if pid not in missing_ids:
            missing_ids.append(pid)

    return new_ids


def process_openpose_frame(frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints, datum.cvOutputData.copy()


def filter_keypoints(keypoints):
    filtered = [p for p in keypoints if sum(1 for kp in p if kp[2] > 0.2) >= 4]
    return np.array(filtered) if filtered else None


def handle_reappearance(pid, person):
    if pid in tracked_persons and tracked_persons[pid]["disappeared"]:
        shoulders = person[2][2] > 0.3 or person[5][2] > 0.3
        nose = person[0][2] > 0.3
        if shoulders or nose:
            print(f"🔄 {pid} reappeared in frame.")
            tracked_persons[pid]["disappeared"] = False
            tracked_persons[pid]["was_disappeared"] = True
            tracked_persons[pid]["disappear_start_time"] = None

            if tracked_persons[pid]["decision"] == "⛔ Disqualified":
                if not tracked_persons[pid].get("beeping_started", False):
                    tracked_persons[pid]["beeping_started"] = True
                    threading.Thread(target=play_beep_loop,
                                     args=(pid,), daemon=True).start()
            else:
                tracked_persons[pid].update({
                    "excluded": False,
                    "baseline_x": None,
                    "baseline_angle": None,
                    "turn_start_time": None,
                    "alerted": False,
                    "decision": "✅ No Action"
                })

            last_alert_time[pid]["hand"] = None
            last_alert_time[pid]["head"] = None
            last_alert_time[pid]["stand"] = None


def is_excluded(pid):
    return pid in tracked_persons and tracked_persons[pid].get("excluded", False)


def safe_broadcast_alert(alert_data):
    """
    Thread-safe function to broadcast alerts from synchronous context
    """
    try:
        # Create new event loop for this thread if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Import here to avoid circular imports
        from man import broadcast_alert

        # Schedule the coroutine
        if loop.is_running():
            # If loop is already running, create a task
            asyncio.create_task(broadcast_alert(alert_data))
        else:
            # If loop is not running, run it
            loop.run_until_complete(broadcast_alert(alert_data))

    except Exception as e:
        print(f"Error broadcasting alert: {e}")


def safe_broadcast_scoreboard(scoreboard_data):
    """
    Thread-safe function to broadcast scoreboard updates
    """
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        from man import broadcast_scoreboard

        if loop.is_running():
            asyncio.create_task(broadcast_scoreboard(scoreboard_data))
        else:
            loop.run_until_complete(broadcast_scoreboard(scoreboard_data))

    except Exception as e:
        print(f"Error broadcasting scoreboard: {e}")


def handle_hand_movement(frame, person, pid, scale, now):
    for name, joint in {"left_hand": person[7], "right_hand": person[4]}.items():
        if joint[2] < 0.4:
            continue
        update_history(pid, name, joint[:2])
        norm = compute_smoothed(pid, name, joint[:2]) / scale

        if last_alert_time[pid]["hand"] and now - last_alert_time[pid]["hand"] < COOLDOWN_PERIOD:
            continue

        if norm >= THRESHOLD:

            img_path = crop_and_save_alert(frame, person, pid, "hand")
            scoreboard[pid] += 1
            decision = evaluate_kbs(pid, scoreboard[pid])
            tracked_persons[pid]["decision"] = decision
            alert = {
                "pid": pid,
                "reason": "hand_movement",
                "decision": decision,
                "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_hand_movement_{screenshot_count-1}.jpg",
                "score": scoreboard[pid]
            }

            # Broadcast alert (thread-safe)
            threading.Thread(target=safe_broadcast_alert,
                             args=(alert,), daemon=True).start()

            # Broadcast updated scoreboard
            threading.Thread(target=safe_broadcast_scoreboard,
                             args=(dict(scoreboard),), daemon=True).start()

            if decision == "⛔ Disqualified" and not tracked_persons[pid].get("notified", False):
                tracked_persons[pid]["excluded"] = True
                print(f"⛔ {pid} has been disqualified.")
                tracked_persons[pid]["notified"] = True
                threading.Thread(target=play_beep_loop,
                                 args=(pid,), daemon=True).start()
            last_alert_time[pid]["hand"] = now


def handle_standing_detection(frame, person, pid, now):
    left, right = person[5], person[2]
    for key in ["left_shoulder_y", "right_shoulder_y"]:
        if pid not in tracking_data:
            tracking_data[pid] = {k: deque(maxlen=HISTORY_LENGTH) for k in [
                "left_hand", "right_hand", "left_shoulder_y", "right_shoulder_y"]}
        elif key not in tracking_data[pid]:
            tracking_data[pid][key] = deque(maxlen=HISTORY_LENGTH)

    if left[2] > 0.4:
        tracking_data[pid]["left_shoulder_y"].append(left[1])
    if right[2] > 0.4:
        tracking_data[pid]["right_shoulder_y"].append(right[1])

    if len(tracking_data[pid]["left_shoulder_y"]) == HISTORY_LENGTH and len(tracking_data[pid]["right_shoulder_y"]) == HISTORY_LENGTH:
        base = max(np.mean(tracking_data[pid]["left_shoulder_y"]), np.mean(
            tracking_data[pid]["right_shoulder_y"]))
        current = max(left[1], right[1])
        rise = base - current

        if rise > 30 and (not last_alert_time[pid].get("stand") or now - last_alert_time[pid]["stand"] > COOLDOWN_PERIOD):
            crop_and_save_alert(frame, person, pid, "standing")
            scoreboard[pid] += 1
            decision = evaluate_kbs(pid, scoreboard[pid])

            tracked_persons[pid]["decision"] = decision
            alert = {
                "pid": pid,
                "reason": "standing",
                "decision": decision,
                "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_standing_{screenshot_count-1}.jpg",
                "score": scoreboard[pid]
            }

            # Broadcast alert
            threading.Thread(target=safe_broadcast_alert,
                             args=(alert,), daemon=True).start()
            threading.Thread(target=safe_broadcast_scoreboard,
                             args=(dict(scoreboard),), daemon=True).start()

            if decision == "⛔ Disqualified" and not tracked_persons[pid].get("notified", False):
                tracked_persons[pid]["excluded"] = True
                print(f"⛔ {pid} has been disqualified.")
                tracked_persons[pid]["notified"] = True
                threading.Thread(target=play_beep_loop,
                                 args=(pid,), daemon=True).start()
            last_alert_time[pid]["stand"] = now
            global id_freeze_until
            global frozen_ids, frozen_positions
            frozen_ids = list(prev_ids)
            frozen_positions = [
                # center between shoulders
                [(p[2][0] + p[5][0]) / 2, (p[2][1] + p[5][1]) / 2]
                for p in prev_people
            ]
            print("📌 Captured frozen positions for ID reassignment.")


def ensure_tracked_person_initialized(pid):
    if pid not in tracked_persons:
        tracked_persons[pid] = {
            "disappear_start_time": None,
            "disappeared": False,
            "was_disappeared": False,
            "excluded": False,
            "baseline_x": None,
            "last_horizontal_alert": None,
            "baseline_angle": None,
            "turn_start_time": None,
            "alerted": False,
            "decision": "✅ No Action"
        }


def handle_horizontal_movement(frame, person, pid, now):
    left, right = person[5], person[2]
    if left[2] < 0.4 or right[2] < 0.4:
        return
    center_x = (left[0] + right[0]) / 2

    if tracked_persons[pid].get("baseline_x") is None:
        tracked_persons[pid]["baseline_x"] = center_x
    else:
        delta_x = abs(center_x - tracked_persons[pid]["baseline_x"])
        if delta_x > 70:
            last_alert = tracked_persons[pid].get("last_horizontal_alert")
            if not last_alert or now - last_alert > timedelta(seconds=5):
                crop_and_save_alert(frame, person, pid, "horizontal_movement")
                scoreboard[pid] += 1
                decision = evaluate_kbs(pid, scoreboard[pid])
                tracked_persons[pid]["decision"] = decision
                alert = {
                    "pid": pid,
                    "reason": "horizontal_movement",
                    "decision": decision,
                    "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_horizontal_movement_{screenshot_count-1}.jpg",
                    
                    "score": scoreboard[pid]
                }

                # Broadcast alert
                threading.Thread(target=safe_broadcast_alert,
                                 args=(alert,), daemon=True).start()
                threading.Thread(target=safe_broadcast_scoreboard, args=(
                    dict(scoreboard),), daemon=True).start()

                if decision == "⛔ Disqualified" and not tracked_persons[pid].get("notified", False):
                    tracked_persons[pid]["excluded"] = True
                    print(f"⛔ {pid} has been disqualified.")
                    tracked_persons[pid]["notified"] = True
                    threading.Thread(target=play_beep_loop,
                                     args=(pid,), daemon=True).start()
                tracked_persons[pid]["last_horizontal_alert"] = now


def evaluate_kbs(pid, score):
    for rule in RULES:
        if rule["min_score"] <= score <= rule["max_score"]:
            return rule["action"]
    return "✅ No Action"


def play_beep_loop_standing(pid):
    while tracked_persons.get(pid, {}).get("standing", False):
        winsound.Beep(1500, 500)  # 1500 Hz, 500 ms
        time.sleep(0.5)


def handle_head_turn_detection(frame, person, pid, now, current_time):
    nose, left_ear, right_ear = person[0], person[17], person[18]
    left_eye, right_eye = person[15], person[16]
    if nose[2] <= 0.1:
        return

    tracked_persons.setdefault(pid, {
        "disappear_start_time": None, "disappeared": False, "nose": tuple(nose[:2]),
        "baseline_angle": None, "turn_start_time": None, "alerted": False, "excluded": False
    })

    if tracked_persons[pid]["excluded"]:
        return

    ref_left = left_ear[0] if left_ear[2] > 0.1 else left_eye[0] if left_eye[2] > 0.1 else None
    ref_right = right_ear[0] if right_ear[2] > 0.1 else right_eye[0] if right_eye[2] > 0.1 else None
    if ref_left is None or ref_right is None:
        return

    center_x = (ref_left + ref_right) / 2
    offset = nose[0] - center_x
    face_width = abs(ref_left - ref_right)
    if face_width == 0:
        return

    angle = (offset / face_width) * 60
    base = tracked_persons[pid]["baseline_angle"]

    if base is None:
        if abs(angle) > MAX_BASELINE_ALLOWED:
            tracked_persons[pid]["excluded"] = True
            return
        tracked_persons[pid]["baseline_angle"] = angle

    rel_angle = angle - tracked_persons[pid]["baseline_angle"]

    frame_width = frame.shape[1]
    edge_margin = 0.1 * frame_width
    if (nose[0] < edge_margin and rel_angle > 0) or (nose[0] > frame_width - edge_margin and rel_angle < 0):
        return

    try:
        sym_ratio = abs(left_ear[0] - nose[0]) / abs(right_ear[0] - nose[0])
    except:
        sym_ratio = None
    sym_ok = sym_ratio is None or sym_ratio < SYM_RATIO_MIN or sym_ratio > SYM_RATIO_MAX

    if abs(rel_angle) > ANGLE_THRESHOLD and abs(rel_angle) > MIN_DEVIATION_FROM_BASELINE and sym_ok:
        if tracked_persons[pid]["turn_start_time"] is None:
            tracked_persons[pid]["turn_start_time"] = current_time
        elif current_time - tracked_persons[pid]["turn_start_time"] >= MIN_HOLD_TIME and not tracked_persons[pid]["alerted"]:
            if not last_alert_time[pid]["head"] or now - last_alert_time[pid]["head"] >= COOLDOWN_PERIOD:
                scoreboard[pid] += 1
                decision = evaluate_kbs(pid, scoreboard[pid])
                tracked_persons[pid]["decision"] = decision
                alert = {
                    "pid": pid,
                    "reason": "head_turn",
                    "decision": decision,
                    "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_head_turn_{screenshot_count-1}.jpg",
                    "score": scoreboard[pid]
                }

                # Broadcast alert
                threading.Thread(target=safe_broadcast_alert,
                                 args=(alert,), daemon=True).start()
                threading.Thread(target=safe_broadcast_scoreboard, args=(
                    dict(scoreboard),), daemon=True).start()

                if decision == "⛔ Disqualified" and not tracked_persons[pid].get("notified", False):
                    tracked_persons[pid]["excluded"] = True
                    print(f"⛔ {pid} has been disqualified.")
                    tracked_persons[pid]["notified"] = True
                    threading.Thread(target=play_beep_loop,
                                     args=(pid,), daemon=True).start()
                crop_and_save_alert(frame, person, pid, "head")
                last_alert_time[pid]["head"] = now
                tracked_persons[pid]["alerted"] = True
    else:
        tracked_persons[pid]["turn_start_time"] = None
        tracked_persons[pid]["alerted"] = False


def draw_disqualified_box(frame, person, pid):
    valid_points = [p[:2] for p in person if p[2] > 0.1]
    if not valid_points:
        return
    points = np.array(valid_points)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    pad = 30
    x, y = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + 2 * pad,
                 frame.shape[1]), min(y + h + 2 * pad, frame.shape[0])
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 3)
    cv2.putText(frame, "⛔ Disqualified", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def mark_unmatched_as_disappeared(current_ids, frame):
    now = time.time()
    for pid, data in tracked_persons.items():
        if pid not in current_ids:
            if data["disappear_start_time"] is None:
                data["disappear_start_time"] = now
            elif not data["disappeared"] and now - data["disappear_start_time"] > 5:
                print(f"🚨 {pid} disappeared from partial scene")
                crop_and_save_alert(frame, frame, pid, "disappear")
                scoreboard[pid] += 1
                decision = evaluate_kbs(pid, scoreboard[pid])
                tracked_persons[pid]["decision"] = decision
                alert = {
                    "pid": pid,
                    "reason": "disappear",
                    "decision": decision,
                    "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_disappear_{screenshot_count-1}.jpg",
                    "score": scoreboard[pid]
                }
                
                # Broadcast alert
                threading.Thread(target=safe_broadcast_alert, args=(alert,), daemon=True).start()
                threading.Thread(target=safe_broadcast_scoreboard, args=(dict(scoreboard),), daemon=True).start()
                
                if decision == "⛔ Disqualified":
                    tracked_persons[pid]["excluded"] = True
                data["disappeared"] = True
        else:
            data["disappear_start_time"] = None  # Reset if matched again


def print_cheating_report(scoreboard):
    if scoreboard:
        print("\n📊 FINAL CHEATING REPORT:")
        for student, score in scoreboard.items():
            print(f"🧑 {student}: {score} cheating points")
    else:
        print("\n📊 No cheating detected during the session.")


def apply_kbs(scoreboard):
    decisions = {}
    for pid, score in scoreboard.items():
        for rule in RULES:
            if rule["min_score"] <= score <= rule["max_score"]:
                decisions[pid] = rule["action"]
                break
        else:
            decisions[pid] = "✅ No Action"
    return decisions


def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


threading.Thread(target=listen_and_transcribe, daemon=True).start()
# Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()


# Update the main import and server start section:
def run_uvicorn():
    """Run the FastAPI server with WebSocket support"""
    import uvicorn
    uvicorn.run("man:app", host="0.0.0.0", port=8000, log_level="info")


# Replace the existing uvicorn thread with:
threading.Thread(target=run_uvicorn, daemon=True).start()

# Add periodic scoreboard broadcasting in your main loop:
last_scoreboard_broadcast = time.time()
SCOREBOARD_BROADCAST_INTERVAL = 5.0  # Broadcast every 5 seconds

# Add this in your main loop, after the frame processing:
# (Add this right before cv2.imshow)

# Periodic scoreboard broadcast
if time.time() - last_scoreboard_broadcast >= SCOREBOARD_BROADCAST_INTERVAL:
    if scoreboard:  # Only broadcast if there are scores
        threading.Thread(target=safe_broadcast_scoreboard,
                         args=(dict(scoreboard),), daemon=True).start()
    last_scoreboard_broadcast = time.time()

# TODO Main Loop
def update_video_streaming(output_frame):
    """Update video frame for streaming to Flutter clients"""
    try:
        # Update the video frame for streaming
        update_video_frame(output_frame)
    except Exception as e:
        print(f"Error updating video frame: {e}")

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
    with frame_lock:
        latest_frame = frame.copy()
        update_video_streaming(latest_frame)

    if current_time - last_process_time >= PROCESS_EVERY_SECONDS:
        last_process_time = current_time

        keypoints, output_frame = process_openpose_frame(frame)

        if keypoints is not None:
            keypoints = filter_keypoints(keypoints)
        else:
            keypoints = None

        if keypoints is not None and keypoints.shape[0] > 0: 
            if time.time() < id_freeze_until:
                ids = prev_ids[:len(keypoints)] if len(prev_ids) >= len(keypoints) else [
                    f"ID_UNKNOWN_{i}" for i in range(len(keypoints))
                ]
                print(
                    "🕒 ID assignment frozen – reusing placeholder IDs (limited tracking)")
            else:
                if frozen_ids and frozen_positions:
                    # Match new people to frozen_positions based on closest shoulder center
                    current_positions = [
                        [(p[2][0] + p[5][0]) / 2, (p[2][1] + p[5][1]) / 2] for p in keypoints
                    ]
                    ids = [None] * len(current_positions)
                    used = set()
                    for i, curr_pos in enumerate(current_positions):
                        min_dist = float("inf")
                        best_j = -1
                        for j, frozen_pos in enumerate(frozen_positions):
                            if j in used:
                                continue
                            dist = np.linalg.norm(
                                np.array(curr_pos) - np.array(frozen_pos))
                            if dist < min_dist:
                                min_dist = dist
                                best_j = j
                        if best_j != -1:
                            ids[i] = frozen_ids[best_j]
                            used.add(best_j)
                    # For unmatched people (new entrants), assign new IDs
                    for i in range(len(ids)):
                        if ids[i] is None:
                            ids[i] = f"ID_{person_id_counter}"
                            person_id_counter += 1
                    print("🔁 Reassigned IDs based on proximity to frozen positions.")
                    frozen_ids = []
                    frozen_positions = []
                else:
                    ids = assign_ids(keypoints, prev_people, prev_ids)
                    # ids = assign_ids_with_faces(keypoints, frame)

                prev_people, prev_ids = keypoints.copy(), ids.copy()

            current_ids = set(ids)
            prev_people, prev_ids = keypoints.copy(), ids.copy()
            # print(prev_ids)
            for i, person in enumerate(keypoints):
                pid = ids[i]
                if tracked_persons.get(pid, {}).get("decision") == "⛔ Disqualified":
                    draw_disqualified_box(output_frame, person, pid)
                ensure_tracked_person_initialized(pid)
                scale = get_body_scale(person)
                if scale < 1e-2:
                    scale = 1.0
                now = datetime.now()

                handle_reappearance(pid, person)
                if is_excluded(pid):
                    continue

                handle_hand_movement(frame, person, pid, scale, now)
                handle_standing_detection(frame, person, pid, now)
                handle_horizontal_movement(frame, person, pid, now)
                handle_head_turn_detection(
                    frame, person, pid, now, current_time)
                if pid in tracked_persons:
                    decision = tracked_persons[pid].get("decision", "")
                    if decision != "✅ No Action":
                        center_x = int((person[5][0] + person[2][0]) / 2)
                        center_y = int((person[5][1] + person[2][1]) / 2) - 40
                        cv2.putText(output_frame, decision, (center_x, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            mark_unmatched_as_disappeared(current_ids, frame)
        else:
            mark_unmatched_as_disappeared(set(), frame)

    if noise_alert_triggered:
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
print_cheating_report(scoreboard)


decisions = apply_kbs(scoreboard)
print("\n📘 FINAL DECISIONS:")
for pid, action in decisions.items():
    print(f"🧑 {pid}: {action}")
