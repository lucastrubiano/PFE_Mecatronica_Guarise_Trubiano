from time import time
from config import CONSECUTIVE_FRAMES_THRESHOLD, TIME_THRESHOLD


class AlertSystem:
    def __init__(self) -> None:
        """
        Initialize Alert System
        """

        # initialize run time
        self.t = 0
        self.last_t = time()

        # initialize counter
        self.consecutive_frames = 0

        # last 10 fatigue predictions
        self.fatigue_prediction = []

        # alert state
        self.alert_state = False

    def run(self, fatigue_prediction) -> None:
        """
        Run Alert System
        """

        # update run time
        self.t = time() - self.last_t
        self.last_t = time()

        # update fatigue prediction vector
        self.__update_history_faigue_prediction(fatigue_prediction)

        # update consecutive frames
        self.__update_consecutive_frames(fatigue_prediction)

        # run alert model
        alert_result = self.alert_model(
            self.t, self.consecutive_frames, self.fatigue_prediction
        )

        # trigger alert
        if alert_result:
            self.alert()

        return alert_result

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

        self.alert_state = True
        print("Alert")
