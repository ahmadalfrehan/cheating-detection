import os
import numpy as np

from collections import deque, defaultdict
from datetime import datetime, timedelta
class Config:
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
    # OBJECT_PASSING_THRESHOLD = 50
    # OBJECT_PASSING_TIME = 1.0

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