import pyopenpose as op  # type: ignore
from flask_socketio import SocketIO, emit
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from man import broadcast_alert, update_video_frame

from threading import Thread, Lock
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
import sounddevice as sd


import sounddevice as sd
from scipy.io.wavfile import write
import whisper


# Configure OpenPose paths.  These environment paths are specific to
# the development machine and may need to be adjusted on other
# systems.  They are retained here from the original code for
# completeness.
os.add_dll_directory(r"C:/openpose/build/x64/Release")
os.add_dll_directory(
    r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
sys.path.append(r"C:/openpose/build/python/openpose/Release")


# --- OpenPose configuration (visual cheating detection) ---
params = {
    "model_folder": "C:/openpose/models",
    "model_pose": "BODY_25",
    "disable_blending": False,
    # Todo remove_this
    "net_resolution": "-1x160",

}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


# Camera configuration.  A high resolution improves the precision of
# OpenPose keypoint localisation.
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# Thresholds and parameters controlling the behaviour of the visual
# detection logic.  These values come directly from the original
# script and are left unchanged.
THRESHOLD = 1.2
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

RISK = "None"

MIN_TOTAL_SCORE = 8

# Knowledge–based system rules mapping total score ranges to actions.
RULES = [
    {"min_score": 1, "max_score": 7, "action": "⚠️ Warning"},
    {"min_score": 8, "max_score": 15, "action": "❗ Under Review"},
    {"min_score": 16, "max_score": float("inf"), "action": "⛔ Disqualified"},
]


# Data structures used by the visual detection logic.  These globals
# track the per‑person history of movements, alerts, and
# identification across frames.
tracking_data: dict = {}

id_freeze_until = 5
frozen_ids: list = []
frozen_positions: list = []  # list of [x, y] shoulder centres

scoreboard: defaultdict[str, int] = defaultdict(int)
prev_people: list = []
prev_ids: list = []
tracked_persons: dict = {}
person_id_counter = 1
last_alert_time = defaultdict(lambda: {"hand": None, "head": None})
screenshot_count = 0
frame_count = 0
last_process_time = 0.0
noise_alert_triggered = False
save_dir = "alerts"
os.makedirs(save_dir, exist_ok=True)

MIC_INDICES = [1, 2]  # Replace with actual device indices


RMS_THRESHOLD = 0.02


SAMPLE_RATE = 16000


CHECK_DURATION = 1


RECORD_DURATION = 10


os.makedirs("recordings", exist_ok=True)

try:
    whisper_model = whisper.load_model("small")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

cheating_keywords = [
    "سؤال", "جواب", "الأول", "ثاني", "ثالث", "رابع", "خامس",
    "ساعدني", "حل", "عطيني", "شو", "الاجابة", "copy", "answer",
    "a", "b", "c", "d", "e", "f",
]

 
latest_frame = None
frame_lock = Lock()


def check_rms(mic_index: int) -> float:
    try:
        data = sd.rec(int(SAMPLE_RATE * CHECK_DURATION),
                      samplerate=SAMPLE_RATE,
                      channels=1,
                      device=mic_index)
        sd.wait()
        rms = np.sqrt(np.mean(np.square(data)))
        return float(rms)
    except Exception as e:
        print(f"❌ Error checking mic {mic_index}: {e}")
        return 0.0


def record_audio(mic_index: int) -> str:
    filename = f"recordings/mic_{mic_index}_{int(time.time())}.wav"
    print(f"🎙 Recording from mic {mic_index} for {RECORD_DURATION} seconds...")
    try:
        data = sd.rec(int(SAMPLE_RATE * RECORD_DURATION),
                      samplerate=SAMPLE_RATE,
                      channels=1,
                      device=mic_index)
        sd.wait()
        write(filename, SAMPLE_RATE, data)
        print(f"💾 Saved recording: {filename}")
        return filename
    except Exception as e:
        print(f"Error recording audio: {e}")
        return ""


def transcribe_audio(filepath: str) -> str:

    print(f"🧠 Transcribing: {filepath}")
    if not whisper_model or not filepath:
        return ""
    try:
        result = whisper_model.transcribe(filepath, language="ar")
        return result.get("text", "").lower().strip()
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""


def detect_keywords(text: str) -> list:

    return [kw for kw in cheating_keywords if kw in text]


def handle_mic(mic_index: int) -> None:
    volume = check_rms(mic_index)
    print(f"🎚 Mic {mic_index} volume: {volume:.4f}")
    if volume > RMS_THRESHOLD:
        print(
            f"🚨 Loud sound detected on mic {mic_index}! Starting recording...")
        filepath = record_audio(mic_index)
        transcript = transcribe_audio(filepath)
        print(f"📄 Transcription (mic {mic_index}): {transcript}")
        keywords = detect_keywords(transcript)
        if keywords:
            print(
                f"🚨 Cheating keywords detected on mic {mic_index}: {', '.join(keywords)}")
            # TODO add socket 
            alert = {
                # "pid": pid,
                "reason": f"Detect In {mic_index}", 
                "decision": f"🚨 Cheating keywords detected on mic {mic_index}: {', '.join(keywords)}",
                # "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_standing_{screenshot_count-1}.jpg",
                # "score": scoreboard[pid]
            }

            # Broadcast alert
            threading.Thread(target=safe_broadcast_alert,
                             args=(alert,), daemon=True).start()
            threading.Thread(target=safe_broadcast_scoreboard,
                             args=(dict(scoreboard),), daemon=True).start()

            try:
                with open("cheating_alerts.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Mic {mic_index}: {', '.join(keywords)}\n")
            except Exception as e:
                print(f"Error writing to alerts file: {e}")
        else:
            print(f"✅ No cheating keywords found for mic {mic_index}.\n")
    else:
        print(f"🔇 No significant sound detected on mic {mic_index}.\n")


def monitor_audio() -> None:

    print("🔁 Monitoring microphones using query method...\n")
    while True:
        for mic in MIC_INDICES:
            handle_mic(mic)

        time.sleep(1)


def listen_for_noise() -> None:

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


def play_beep_loop(pid: str) -> None:

    while tracked_persons.get(pid, {}).get("decision") == "⛔ Disqualified" and not tracked_persons[pid].get("disappeared", False):
        winsound.Beep(1000, 500)
        time.sleep(0.5)


def get_body_scale(person: np.ndarray) -> float:

    if person[5][2] > 0.4 and person[2][2] > 0.4:
        return float(np.linalg.norm(person[5][:2] - person[2][:2]))
    return 1.0


def update_history(pid: str, joint: str, pt: np.ndarray) -> None:

    if pid not in tracking_data:
        tracking_data[pid] = {
            "left_hand": deque(maxlen=HISTORY_LENGTH),
            "right_hand": deque(maxlen=HISTORY_LENGTH),
            "left_shoulder_y": deque(maxlen=HISTORY_LENGTH),
            "right_shoulder_y": deque(maxlen=HISTORY_LENGTH),
        }
    tracking_data[pid][joint].append(pt)


def compute_smoothed(pid: str, joint: str, pt: np.ndarray) -> float:

    history = tracking_data[pid][joint]
    if not history:
        return 0.0
    avg = np.mean(history, axis=0)
    return float(np.linalg.norm(pt - avg))


def crop_and_save_alert(frame: np.ndarray, person: np.ndarray, pid: str, reason: str) -> None:

    global screenshot_count, RISK

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
    print(f"⚠️ {pid} {reason} alert -> {filename}, Risk for movement is: \"{RISK}\"")
    screenshot_count += 1


# In the original code an alternate assign_ids function was commented out.
# Here we keep only the improved assign_ids function with re‑use of
# disappeared person identifiers.
missing_ids = deque()  # Track identifiers of people who have disappeared


def assign_ids(current: np.ndarray, previous: np.ndarray, previous_ids: list, threshold: float = 1000.0) -> list:
    """Assign persistent identifiers to detected people based on proximity.

    This function matches new people to previously seen individuals by
    comparing neck positions.  When a person disappears, their ID
    becomes available for re‑use.  New entrants receive a fresh ID.
    """
    global person_id_counter, missing_ids
    new_ids: list = [None] * len(current)
    used = set()
    current_active_ids = set()
    for i, p in enumerate(current):
        best_dist = float("inf")
        match = -1
        for j, q in enumerate(previous):
            if j in used:
                continue
            # Skip if either neck is low confidence
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
    previous_id_set = set(previous_ids)
    disappeared = previous_id_set - current_active_ids
    for pid in disappeared:
        if pid not in missing_ids:
            missing_ids.append(pid)
    return new_ids


def process_openpose_frame(frame: np.ndarray) -> tuple:
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints, datum.cvOutputData.copy()


def filter_keypoints(keypoints: np.ndarray) -> np.ndarray | None:
    filtered = [p for p in keypoints if sum(1 for kp in p if kp[2] > 0.2) >= 4]
    return np.array(filtered) if filtered else None


def handle_reappearance(pid: str, person: np.ndarray) -> None:

    if pid in tracked_persons and tracked_persons[pid]["disappeared"]:
        shoulders = person[2][2] > 0.3 or person[5][2] > 0.3
        nose = person[0][2] > 0.3
        if shoulders or nose:
            print(f"🔄 {pid} reappeared in frame.")
            tracked_persons[pid]["disappeared"] = False
            tracked_persons[pid]["was_disappeared"] = True
            tracked_persons[pid]["disappear_start_time"] = None
            # If the person was previously disqualified, restart the beep loop
            alert = {
                "pid": pid,
                "reason": "head_turn",
                "risk":RISK,
                "decision": '',
                "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_head_turn_{screenshot_count-1}.jpg",
                "score": scoreboard[pid]
            }
                # Broadcast alert
            threading.Thread(target=safe_broadcast_alert,
                                 args=(alert,), daemon=True).start()
            threading.Thread(target=safe_broadcast_scoreboard, args=(
                    dict(scoreboard),), daemon=True).start()

                
            if tracked_persons[pid]["decision"] == "⛔ Disqualified":
                if not tracked_persons[pid].get("beeping_started", False):
                    tracked_persons[pid]["beeping_started"] = True
                    threading.Thread(target=play_beep_loop,
                                     args=(pid,), daemon=True).start()
            else:
                # Reset state for a valid participant
                tracked_persons[pid].update({
                    "excluded": False,
                    "baseline_x": None,
                    "baseline_angle": None,
                    "turn_start_time": None,
                    "alerted": False,
                    "decision": "✅ No Action",
                })
            # Reset alert timestamps
            last_alert_time[pid]["hand"] = None
            last_alert_time[pid]["head"] = None
            last_alert_time[pid]["stand"] = None


def is_excluded(pid: str) -> bool:

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


def handle_hand_movement(frame: np.ndarray, person: np.ndarray, pid: str, scale: float, now: datetime) -> None:
    global RISK
    RISK = "Low"

    for name, joint in {"left_hand": person[7], "right_hand": person[4]}.items():
        if joint[2] < 0.4:
            continue
        update_history(pid, name, joint[:2])
        norm = compute_smoothed(pid, name, joint[:2]) / scale
        if last_alert_time[pid]["hand"] and now - last_alert_time[pid]["hand"] < COOLDOWN_PERIOD:
            continue
        if norm >= THRESHOLD:
            crop_and_save_alert(frame, person, pid, "hand")
            scoreboard[pid] += 1

            decision = evaluate_kbs(pid, scoreboard[pid])
            tracked_persons[pid]["decision"] = decision
            tracked_persons[pid]["risk"] = "Low"
            alert = {
                "pid": pid,
                "risk": RISK,
                "reason": "hand_movement",
                "decision": decision,
                "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_hand_{screenshot_count-1}.jpg",
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


def handle_standing_detection(frame: np.ndarray, person: np.ndarray, pid: str, now: datetime) -> None:
    global RISK
    RISK = "High"

    """Detect when a person stands up by monitoring shoulder heights."""
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
            scoreboard[pid] += 3
            tracked_persons[pid]["risk"] = "High"
            decision = evaluate_kbs(pid, scoreboard[pid])
            tracked_persons[pid]["decision"] = decision
            alert = {
                "pid": pid,

                "reason": "standing",
                "risk": RISK,
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
            global id_freeze_until, frozen_ids, frozen_positions
            frozen_ids = list(prev_ids)
            frozen_positions = [
                # centre between shoulders
                [(p[2][0] + p[5][0]) / 2, (p[2][1] + p[5][1]) / 2]
                for p in prev_people
            ]
            print("📌 Captured frozen positions for ID reassignment.")


def ensure_tracked_person_initialized(pid: str) -> None:
    """Initialise default state for a tracked person if needed."""
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
            "decision": "✅ No Action",
            "risk": "None",
        }


def handle_horizontal_movement(frame: np.ndarray, person: np.ndarray, pid: str, now: datetime) -> None:
    global RISK
    RISK = "Low"

    """Detect large horizontal movements (leaning or shifting).

    When the centre of the shoulders drifts too far from its baseline
    value, an alert is triggered.  This helps catch movements like
    leaning out of the field of view to look at another monitor.
    """
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
                tracked_persons[pid]["risk"] = "Low"
                decision = evaluate_kbs(pid, scoreboard[pid])
                tracked_persons[pid]["decision"] = decision
                alert = {
                    "pid": pid,
                    "risk": RISK,
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


def evaluate_kbs(pid: str, score: int) -> str:
    """Evaluate the knowledge‑based system and return the decision text."""
    for rule in RULES:
        if rule["min_score"] <= score <= rule["max_score"]:
            return rule["action"]
    return "✅ No Action"


def play_beep_loop_standing(pid: str) -> None:
    """Emit a beep while a person remains standing.  This is unused in
    the current implementation but retained for completeness."""
    while tracked_persons.get(pid, {}).get("standing", False):
        winsound.Beep(1500, 500)
        time.sleep(0.5)


def handle_head_turn_detection(frame: np.ndarray, person: np.ndarray, pid: str, now: datetime, current_time: float) -> None:
    global RISK
    RISK = "Medium"

    """Detect when a person turns their head away from their baseline direction."""
    nose, left_ear, right_ear = person[0], person[17], person[18]
    left_eye, right_eye = person[15], person[16]
    if nose[2] <= 0.1:
        return
    # Initialise state for this person
    tracked_persons.setdefault(pid, {
        "disappear_start_time": None,
        "disappeared": False,
        "nose": tuple(nose[:2]),
        "baseline_angle": None,
        "turn_start_time": None,
        "alerted": False,
        "excluded": False,
    })
    if tracked_persons[pid]["excluded"]:
        return
    # Determine reference points for face symmetry (ears or eyes)
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
    except Exception:
        sym_ratio = None
    sym_ok = sym_ratio is None or sym_ratio < SYM_RATIO_MIN or sym_ratio > SYM_RATIO_MAX
    if abs(rel_angle) > ANGLE_THRESHOLD and abs(rel_angle) > MIN_DEVIATION_FROM_BASELINE and sym_ok:
        if tracked_persons[pid]["turn_start_time"] is None:
            tracked_persons[pid]["turn_start_time"] = current_time
        elif current_time - tracked_persons[pid]["turn_start_time"] >= MIN_HOLD_TIME and not tracked_persons[pid]["alerted"]:
            if not last_alert_time[pid]["head"] or now - last_alert_time[pid]["head"] >= COOLDOWN_PERIOD:
                scoreboard[pid] += 2
                tracked_persons[pid]["risk"] = "Medium"

                decision = evaluate_kbs(pid, scoreboard[pid])
                tracked_persons[pid]["decision"] = decision
                alert = {
                    "pid": pid,
                    "reason": "head_turn",
                    "risk": RISK,
                    "decision": decision,
                    # "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_horizontal_movement_{screenshot_count-1}.jpg",
                    
                    "screenshot_url": f"http://127.0.0.1:8000/alerts/alert_{pid}_head_{screenshot_count-1}.jpg",
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


def draw_disqualified_box(frame: np.ndarray, person: np.ndarray, pid: str) -> None:
    """Draw a red bounding box and label around a disqualified person."""
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


def mark_unmatched_as_disappeared(current_ids: set, frame: np.ndarray) -> None:
    """Identify persons who have disappeared from view and update their state."""
    now_time = time.time()
    for pid, data in tracked_persons.items():
        if pid not in current_ids:
            if data["disappear_start_time"] is None:
                data["disappear_start_time"] = now_time
            elif not data["disappeared"] and now_time - data["disappear_start_time"] > 5:
                print(f"🚨 {pid} disappeared from partial scene")
                crop_and_save_alert(frame, frame, pid, "disappear")
                scoreboard[pid] += 1
                decision = evaluate_kbs(pid, scoreboard[pid])
                tracked_persons[pid]["decision"] = decision
                if decision == "⛔ Disqualified":
                    tracked_persons[pid]["excluded"] = True
                data["disappeared"] = True
        else:
            data["disappear_start_time"] = None


def print_cheating_report(scoreboard: dict) -> None:
    """Print a summary of cheating points accumulated by each student."""
    if scoreboard:
        print("\n📊 FINAL CHEATING REPORT:")
        for student, score in scoreboard.items():
            print(f"🧑 {student}: {score} cheating points")
    else:
        print("\n📊 No cheating detected during the session.")


def apply_kbs(scoreboard: dict) -> dict:
    """Apply the KBS rules to each score and return final decisions."""
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


# threading.Thread(target=listen_and_transcribe, daemon=True).start()
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




def main() -> None:
    """Main entry point for the integrated cheating detection system."""
    global frame_count, last_process_time, prev_people, prev_ids, noise_alert_triggered, person_id_counter, frozen_ids, frozen_positions
    # Launch the audio monitoring thread.  It runs as a daemon so it
    # will exit when the main program terminates.
    threading.Thread(target=monitor_audio, daemon=True).start()
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
        output_frame = frame.copy()
        current_time = time.time()
        with frame_lock:  
            latest_frame = frame.copy()
            update_video_streaming(latest_frame)
 
        # Periodically run OpenPose processing based on PROCESS_EVERY_SECONDS
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
                        f"ID_UNKNOWN_{i}" for i in range(len(keypoints))]
                    print(
                        "🕒 ID assignment frozen – reusing placeholder IDs (limited tracking)")
                else:
                    if frozen_ids and frozen_positions:
                        # Match new people to frozen positions based on closest shoulder centres
                        current_positions = [
                            [(p[2][0] + p[5][0]) / 2, (p[2][1] + p[5][1]) / 2]
                            for p in keypoints
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
                        print(
                            "🔁 Reassigned IDs based on proximity to frozen positions.")
                        # Clear frozen state
                        frozen_ids = []
                        frozen_positions = []
                    else:
                        ids = assign_ids(keypoints, prev_people, prev_ids)
                prev_people, prev_ids = keypoints.copy(), ids.copy()
                current_ids = set(ids)
                # Update state for each detected person
                prev_people, prev_ids = keypoints.copy(), ids.copy()
                for i, person in enumerate(keypoints):
                    pid = ids[i]
                    # Draw a special box if the person has been disqualified
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
                        risk = tracked_persons[pid].get("risk", "None")
                        if decision != "✅ No Action":
                            center_x = int((person[5][0] + person[2][0]) / 2)
                            center_y = int(
                                (person[5][1] + person[2][1]) / 2) - 40
                            cv2.putText(output_frame, decision, (center_x, center_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv2.putText(output_frame, f"Risk: {risk}", (
                                center_x, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                mark_unmatched_as_disappeared(current_ids, frame)
            else:
                mark_unmatched_as_disappeared(set(), frame)
        # Show noise alert if triggered
        if noise_alert_triggered:
            cv2.putText(output_frame, "🚨 Noise Detected!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            noise_alert_triggered = False
        # Display FPS counter
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


if __name__ == "__main__":
    main()
