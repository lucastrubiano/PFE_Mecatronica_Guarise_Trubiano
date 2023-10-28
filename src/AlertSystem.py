from __future__ import annotations

import threading
from time import time

import RPi.GPIO as GPIO

from config import CONSECUTIVE_FRAMES_THRESHOLD
from config import CONSECUTIVE_SEC_ALERT_THRESHOLD
from config import FPS
from config import LED_FATIGUE
from config import STATE_DISTRACTED
from config import STATE_FATIGUE
from config import STATE_NORMAL
from config import THRESHOLD_EAR
from config import THRESHOLD_TIME


GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_FATIGUE, GPIO.OUT)


class AlertSystem():
    def __init__(self) -> None:
        """
        Initialize Alert System
        """

        # initialize run time
        self.last_t = time()

    def run(self, state_prediction: str) -> None:
        """
        Run Alert System
        """

        if state_prediction == STATE_FATIGUE:
            # make a beep sound in parallel
            led_thread = threading.Thread(
                target=self.alert, args=(LED_FATIGUE),
            )
            led_thread.start()
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

        if t > THRESHOLD_TIME and consecutive_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
            alert_result = True

        return alert_result

    def alert(self, led_pin: str, buzzer: bool = True) -> None:
        """
        Alert. Turn on buzzer and led light in raspberry pi.
        Possible in a future will be implemented a sms or email alert.
        """

        try:
            if (self.last_t - time()) > CONSECUTIVE_SEC_ALERT_THRESHOLD:
                GPIO.output(led_pin, GPIO.HIGH)
        except KeyboardInterrupt:
            # If you press Ctrl+C, this block will clean up the GPIO settings
            GPIO.cleanup()
