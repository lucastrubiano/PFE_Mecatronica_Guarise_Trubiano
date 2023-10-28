from __future__ import annotations

import threading
from time import time

import RPi.GPIO as GPIO

from config import CONSECUTIVE_FRAMES_THRESHOLD
from config import CONSECUTIVE_SEC_ALERT_THRESHOLD
from config import FPS
from config import PIN_FATIGUE
from config import PIN_BUSSER
from config import PIN_DISTRACTED
from config import STATE_DISTRACTED
from config import STATE_FATIGUE
from config import STATE_NORMAL
from config import THRESHOLD_EAR
from config import THRESHOLD_TIME
from config import PIN_STATUS_LOW
from config import PIN_STATUS_HIGH



GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN_FATIGUE, GPIO.OUT)
GPIO.setup(PIN_BUSSER, GPIO.OUT)
GPIO.setup(PIN_DISTRACTED, GPIO.OUT)



class AlertSystem():
    def __init__(self) -> None:
        """
        Initialize Alert System
        """

        GPIO.output(PIN_FATIGUE, GPIO.LOW)
        self.last_t = time()

    def run(self, state_prediction: str) -> None:
        """
        Run Alert System
        """

        self.alert(state_prediction)


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
    
    def turn_on_off_device(self, led_pin, status):

        try:
            if status == PIN_STATUS_HIGH:
                GPIO.output(led_pin, GPIO.HIGH)
            elif status == PIN_STATUS_LOW:
                GPIO.output(led_pin, GPIO.LOW)
            else:
                GPIO.output(led_pin, GPIO.LOW)
        except KeyboardInterrupt:
            # If you press Ctrl+C, this block will clean up the GPIO settings
            GPIO.output(led_pin, GPIO.LOW)
            GPIO.cleanup()

    def alert(self, state_prediction: str) -> None:
        """
        Alert. Turn on buzzer and led light in raspberry pi.
        Possible in a future will be implemented a sms or email alert.
        """

        if state_prediction == STATE_FATIGUE:
            self.last_t = time()
            self.turn_on_off_device(PIN_FATIGUE, PIN_STATUS_HIGH)
            self.turn_on_off_device(PIN_BUSSER, PIN_STATUS_HIGH)

        if state_prediction == STATE_DISTRACTED:
            self.last_t = time()
            self.turn_on_off_device(PIN_DISTRACTED, PIN_STATUS_HIGH)

        if state_prediction != STATE_FATIGUE:
            if (time() - self.last_t) > CONSECUTIVE_SEC_ALERT_THRESHOLD:
                self.turn_on_off_device(PIN_FATIGUE, PIN_STATUS_LOW)
                self.turn_on_off_device(PIN_BUSSER, PIN_STATUS_LOW)
        
        if state_prediction != STATE_DISTRACTED:
            if (time() - self.last_t) > CONSECUTIVE_SEC_ALERT_THRESHOLD:
                self.turn_on_off_device(PIN_DISTRACTED, PIN_STATUS_LOW)