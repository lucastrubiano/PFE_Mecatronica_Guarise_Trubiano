from __future__ import annotations

import threading
from time import time

from config import CONSECUTIVE_FRAMES_THRESHOLD
from config import TIME_THRESHOLD


class AlertSystem(threading.Thread):
    def __init__(self) -> None:
        """
        Initialize Alert System
        """
        threading.Thread.__init__(self)

        # initialize run time
        self.t = 0
        self.last_t = time()

        # initialize counter
        # self.consecutive_frames = 0

        # last 10 fatigue predictions
        # self.fatigue_prediction = []

        # alert state
        self.alert_state = False

        self.playing_sound = False

    def run(self, fatigue_prediction) -> None:
        """
        Run Alert System
        """

        if fatigue_prediction > 0.8:
            # make a beep sound in parallel
            self.alert()
            alert = 1

        else:
            alert = 0

        return alert

    def __update_consecutive_frames(self, fatigue_prediction) -> None:
        """
        Update consecutive frames
        """

        if fatigue_prediction == 1:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0

    def __update_history_faigue_prediction(self, fatigue_prediction) -> None:
        """
        Update history fatigue prediction. Only the last 10 fatigue predictions are stored.
        """

        self.fatigue_prediction.append(fatigue_prediction)

        if len(self.fatigue_prediction) > 10:
            self.fatigue_prediction.pop(0)

    def alert_model(self, t, consecutive_frames, fatigue_prediction) -> bool:
        """
        Alert Model
        """

        alert_result = False

        if t > TIME_THRESHOLD and consecutive_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
            alert_result = True

        return alert_result

    def alert(self) -> None:
        """
        Alert. Turn on buzzer and led light in raspberry pi.
        Possible in a future will be implemented a sms or email alert.
        """

        if not self.playing_sound and self.last_t + 5 < time():
            self.last_t = time()
            self.playing_sound = True
            self.playing_sound = False
