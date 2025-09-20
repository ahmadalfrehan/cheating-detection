"""
This module combines a visual cheating‑detection system with an
independent audio‑based speech recognition monitor.  The visual
component uses OpenPose to track people in a live camera feed and
generates alerts when hands move unexpectedly, a person stands up,
turns their head for too long, or walks out of frame.  Each alert
increments a per‑person score and a simple knowledge‑based system
maps accumulated scores to actions such as warnings, under review or
disqualification.  A separate microphone monitor runs on its own
thread; it listens for loud noises, records a short audio clip,
transcribes it using a Whisper model, and logs any suspicious
keywords to a text file.  The two subsystems operate independently so
that neither interferes with the other.

The original visual detection logic has been preserved exactly as
requested.  Only the speech recognition functions have been added
alongside it, and a thread is launched at startup to perform the
audio monitoring.  Nothing from the core cheating‑detection code has
been removed or altered.
"""

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


# --- BEGIN Imports for speech recognition functionality ---
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
# --- END Imports for speech recognition functionality ---


# Configure OpenPose paths.  These environment paths are specific to
# the development machine and may need to be adjusted on other
# systems.  They are retained here from the original code for
# completeness.
os.add_dll_directory(r"C:/openpose/build/x64/Release")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
sys.path.append(r"C:/openpose/build/python/openpose/Release")
from openpose import pyopenpose as op  # type: ignore


# --- OpenPose configuration (visual cheating detection) ---
params = {
    "model_folder": "C:/openpose/models",
    "model_pose": "BODY_25",
    "disable_blending": False,
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


# --- Audio monitoring configuration and globals ---

# Indices of the microphones to monitor.  These may need to be
# adjusted depending on the system.  Use ``sd.query_devices()`` to
# determine the correct indices.  Multiple microphones can be
# monitored simultaneously.
MIC_INDICES = [1, 2]  # Replace with actual device indices

# Root‑mean‑square (RMS) amplitude threshold above which the system
# will begin recording audio.  A lower threshold makes the system
# more sensitive to background noise.
RMS_THRESHOLD = 0.02

# Sample rate for recording audio.
SAMPLE_RATE = 16000

# Duration (in seconds) to measure the RMS volume when deciding
# whether to record.
CHECK_DURATION = 1

# Duration (in seconds) of each audio recording.  Longer recordings
# may improve transcription quality but will delay subsequent
# detections.
RECORD_DURATION = 10

# Ensure the folder used to save recordings exists.
os.makedirs("recordings", exist_ok=True)

# Load the Whisper model once at startup.  Loading can be slow, so
# doing it here prevents repeated initialisation overhead inside the
# monitoring thread.  Use a smaller model if GPU memory is limited.
try:
    whisper_model = whisper.load_model("medium")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

# Keywords (Arabic and English) that are considered indicative of
# cheating when detected in the transcribed audio.
cheating_keywords = [
    "سؤال", "جواب", "الأول", "ثاني", "ثالث", "رابع", "خامس",
    "ساعدني", "حل", "عطيني", "شو", "الاجابة", "copy", "answer",
    "a", "b", "c", "d", "e", "f",
]


def check_rms(mic_index: int) -> float:
    """Check the RMS volume level for the given microphone index.

    A short snippet of audio is recorded from the specified device and
    its root‑mean‑square amplitude is computed.  If an error occurs
    while reading from the device, zero is returned.  The function
    blocks for ``CHECK_DURATION`` seconds.
    """
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
    """Record audio from the specified microphone for a fixed duration.

    A file named ``recordings/mic_{mic_index}_{timestamp}.wav`` is
    created in the ``recordings`` directory.  If recording fails, an
    empty string is returned.
    """
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
    """Use Whisper to transcribe recorded audio to text.

    If transcription fails or the model is not available, an empty
    string is returned.  The returned text is lower‑cased and stripped.
    """
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
    """Return a list of cheating keywords found in the given text.

    The check is case‑insensitive; the input should already be
    lower‑cased.  If no keywords are present, an empty list is
    returned.
    """
    return [kw for kw in cheating_keywords if kw in text]


def handle_mic(mic_index: int) -> None:
    """Main logic per mic: check sound > threshold → record → transcribe → analyse.

    For each microphone, the RMS amplitude is measured.  If it exceeds
    ``RMS_THRESHOLD`` a recording is made, transcribed, and searched
    for cheating keywords.  Any detections are appended to the log
    file ``cheating_alerts.txt`` with a timestamp.
    """
    volume = check_rms(mic_index)
    print(f"🎚 Mic {mic_index} volume: {volume:.4f}")
    if volume > RMS_THRESHOLD:
        print(f"🚨 Loud sound detected on mic {mic_index}! Starting recording...")
        filepath = record_audio(mic_index)
        transcript = transcribe_audio(filepath)
        print(f"📄 Transcription (mic {mic_index}): {transcript}")
        keywords = detect_keywords(transcript)
        if keywords:
            print(f"🚨 Cheating keywords detected on mic {mic_index}: {', '.join(keywords)}")
            try:
                with open("cheating_alerts.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Mic {mic_index}: {', '.join(keywords)}\n")
            except Exception as e:
                print(f"Error writing to alerts file: {e}")
        else:
            print(f"✅ No cheating keywords found for mic {mic_index}.\n")
    else:
        print(f"🔇 No significant sound detected on mic {mic_index}.\n")


def monitor_audio() -> None:
    """Continuously monitor all configured microphones and transcribe any speech.

    This function runs in its own thread so that audio monitoring does not
    block the main visual detection loop.  It iterates over each
    configured microphone index, processes it via ``handle_mic``, and
    sleeps for one second between iterations to avoid excessive CPU
    usage.
    """
    print("🔁 Monitoring microphones using query method...\n")
    while True:
        for mic in MIC_INDICES:
            handle_mic(mic)
        # Sleep to reduce CPU usage and align with CHECK_DURATION
        time.sleep(1)


# --- Visual cheating detection functions (unchanged from original) ---

def listen_for_noise() -> None:
    """Continuously sample ambient noise from the default input device
    using PyAudio and flag when the RMS surpasses ``NOISE_THRESHOLD``.

    This function runs in a separate daemon thread to avoid blocking
    the main loop.
    """
    global noise_alert_triggered
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        rms = audioop.rms(data, 2)
        if rms > NOISE_THRESHOLD:
            noise_alert_triggered = True


# Start the ambient noise listener thread immediately.  This thread
# monitors for generic loud noises and sets a global flag when
# triggered.  It is preserved from the original code.
threading.Thread(target=listen_for_noise, daemon=True).start()


def play_beep_loop(pid: str) -> None:
    """Emit a periodic beep when a person is disqualified.

    The loop continues until the tracked person is no longer
    considered disqualified or has disappeared entirely.  A higher
    frequency beep denotes disqualification.
    """
    while tracked_persons.get(pid, {}).get("decision") == "⛔ Disqualified" and not tracked_persons[pid].get("disappeared", False):
        winsound.Beep(1000, 500)
        time.sleep(0.5)


def get_body_scale(person: np.ndarray) -> float:
    """Return a rough measure of body scale based on shoulder width.

    The scale is used to normalise hand movement thresholds across
    different distances to the camera.  If shoulder visibility is
    poor, a default scale of 1.0 is returned.
    """
    if person[5][2] > 0.4 and person[2][2] > 0.4:
        return float(np.linalg.norm(person[5][:2] - person[2][:2]))
    return 1.0


def update_history(pid: str, joint: str, pt: np.ndarray) -> None:
    """Update the movement history deque for a given person and joint."""
    if pid not in tracking_data:
        tracking_data[pid] = {
            "left_hand": deque(maxlen=HISTORY_LENGTH),
            "right_hand": deque(maxlen=HISTORY_LENGTH),
            "left_shoulder_y": deque(maxlen=HISTORY_LENGTH),
            "right_shoulder_y": deque(maxlen=HISTORY_LENGTH),
        }
    tracking_data[pid][joint].append(pt)


def compute_smoothed(pid: str, joint: str, pt: np.ndarray) -> float:
    """Compute the norm of the difference between the current point
    ``pt`` and the historical average for a given joint.  Returns zero
    if no history exists.
    """
    history = tracking_data[pid][joint]
    if not history:
        return 0.0
    avg = np.mean(history, axis=0)
    return float(np.linalg.norm(pt - avg))


def crop_and_save_alert(frame: np.ndarray, person: np.ndarray, pid: str, reason: str) -> None:
    """Crop an image around a person and save it to disk with a reason."""
    global screenshot_count
    # Ensure we have valid keypoints to compute a bounding box.  If
    # there are no valid points the entire frame is saved as a fallback.
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
    """Run OpenPose on a frame and return keypoints and the annotated image."""
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints, datum.cvOutputData.copy()


def filter_keypoints(keypoints: np.ndarray) -> np.ndarray | None:
    """Filter out person detections with fewer than four confident keypoints."""
    filtered = [p for p in keypoints if sum(1 for kp in p if kp[2] > 0.2) >= 4]
    return np.array(filtered) if filtered else None


def handle_reappearance(pid: str, person: np.ndarray) -> None:
    """Reset a person's state when they reappear in the frame."""
    if pid in tracked_persons and tracked_persons[pid]["disappeared"]:
        shoulders = person[2][2] > 0.3 or person[5][2] > 0.3
        nose = person[0][2] > 0.3
        if shoulders or nose:
            print(f"🔄 {pid} reappeared in frame.")
            tracked_persons[pid]["disappeared"] = False
            tracked_persons[pid]["was_disappeared"] = True
            tracked_persons[pid]["disappear_start_time"] = None
            # If the person was previously disqualified, restart the beep loop
            if tracked_persons[pid]["decision"] == "⛔ Disqualified":
                if not tracked_persons[pid].get("beeping_started", False):
                    tracked_persons[pid]["beeping_started"] = True
                    threading.Thread(target=play_beep_loop, args=(pid,), daemon=True).start()
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
    """Return True if the person is excluded from further checks."""
    return pid in tracked_persons and tracked_persons[pid].get("excluded", False)


def handle_hand_movement(frame: np.ndarray, person: np.ndarray, pid: str, scale: float, now: datetime) -> None:
    """Detect significant hand movement relative to a smoothed baseline."""
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
            if decision == "⛔ Disqualified" and not tracked_persons[pid].get("notified", False):
                tracked_persons[pid]["excluded"] = True
                print(f"⛔ {pid} has been disqualified.")
                tracked_persons[pid]["notified"] = True
                threading.Thread(target=play_beep_loop, args=(pid,), daemon=True).start()
            last_alert_time[pid]["hand"] = now


def handle_standing_detection(frame: np.ndarray, person: np.ndarray, pid: str, now: datetime) -> None:
    """Detect when a person stands up by monitoring shoulder heights."""
    left, right = person[5], person[2]
    for key in ["left_shoulder_y", "right_shoulder_y"]:
        if pid not in tracking_data:
            tracking_data[pid] = {k: deque(maxlen=HISTORY_LENGTH) for k in ["left_hand", "right_hand", "left_shoulder_y", "right_shoulder_y"]}
        elif key not in tracking_data[pid]:
            tracking_data[pid][key] = deque(maxlen=HISTORY_LENGTH)
    if left[2] > 0.4:
        tracking_data[pid]["left_shoulder_y"].append(left[1])
    if right[2] > 0.4:
        tracking_data[pid]["right_shoulder_y"].append(right[1])
    if len(tracking_data[pid]["left_shoulder_y"]) == HISTORY_LENGTH and len(tracking_data[pid]["right_shoulder_y"]) == HISTORY_LENGTH:
        base = max(np.mean(tracking_data[pid]["left_shoulder_y"]), np.mean(tracking_data[pid]["right_shoulder_y"]))
        current = max(left[1], right[1])
        rise = base - current
        if rise > 30 and (not last_alert_time[pid].get("stand") or now - last_alert_time[pid]["stand"] > COOLDOWN_PERIOD):
            crop_and_save_alert(frame, person, pid, "standing")
            scoreboard[pid] += 3
            decision = evaluate_kbs(pid, scoreboard[pid])
            tracked_persons[pid]["decision"] = decision
            if decision == "⛔ Disqualified" and not tracked_persons[pid].get("notified", False):
                tracked_persons[pid]["excluded"] = True
                print(f"⛔ {pid} has been disqualified.")
                tracked_persons[pid]["notified"] = True
                threading.Thread(target=play_beep_loop, args=(pid,), daemon=True).start()
            last_alert_time[pid]["stand"] = now
            global id_freeze_until, frozen_ids, frozen_positions
            frozen_ids = list(prev_ids)
            frozen_positions = [
                [(p[2][0] + p[5][0]) / 2, (p[2][1] + p[5][1]) / 2]  # centre between shoulders
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
        }


def handle_horizontal_movement(frame: np.ndarray, person: np.ndarray, pid: str, now: datetime) -> None:
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
                decision = evaluate_kbs(pid, scoreboard[pid])
                tracked_persons[pid]["decision"] = decision
                if decision == "⛔ Disqualified" and not tracked_persons[pid].get("notified", False):
                    tracked_persons[pid]["excluded"] = True
                    print(f"⛔ {pid} has been disqualified.")
                    tracked_persons[pid]["notified"] = True
                    threading.Thread(target=play_beep_loop, args=(pid,), daemon=True).start()
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
                decision = evaluate_kbs(pid, scoreboard[pid])
                tracked_persons[pid]["decision"] = decision
                if decision == "⛔ Disqualified" and not tracked_persons[pid].get("notified", False):
                    tracked_persons[pid]["excluded"] = True
                    print(f"⛔ {pid} has been disqualified.")
                    tracked_persons[pid]["notified"] = True
                    threading.Thread(target=play_beep_loop, args=(pid,), daemon=True).start()
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
    x2, y2 = min(x + w + 2 * pad, frame.shape[1]), min(y + h + 2 * pad, frame.shape[0])
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 3)
    cv2.putText(frame, "⛔ Disqualified", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


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
                    ids = prev_ids[:len(keypoints)] if len(prev_ids) >= len(keypoints) else [f"ID_UNKNOWN_{i}" for i in range(len(keypoints))]
                    print("🕒 ID assignment frozen – reusing placeholder IDs (limited tracking)")
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
                                dist = np.linalg.norm(np.array(curr_pos) - np.array(frozen_pos))
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
                    handle_head_turn_detection(frame, person, pid, now, current_time)
                    if pid in tracked_persons:
                        decision = tracked_persons[pid].get("decision", "")
                        if decision != "✅ No Action":
                            center_x = int((person[5][0] + person[2][0]) / 2)
                            center_y = int((person[5][1] + person[2][1]) / 2) - 40
                            cv2.putText(output_frame, decision, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                mark_unmatched_as_disappeared(current_ids, frame)
            else:
                mark_unmatched_as_disappeared(set(), frame)
        # Show noise alert if triggered
        if noise_alert_triggered:
            cv2.putText(output_frame, "🚨 Noise Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            noise_alert_triggered = False
        # Display FPS counter
        cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
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
