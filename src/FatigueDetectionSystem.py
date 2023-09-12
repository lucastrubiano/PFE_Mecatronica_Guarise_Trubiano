from scipy.spatial import distance as dist

from config import EAR_THRESHOLD, MAR_THRESHOLD

from datetime import datetime


class FatigueDetectionSystem:
    def __init__(self) -> None:
        """
        Initialize Fatigue Detection System
        """

        # Set initial values
        self.avg_ear = 0
        self.avg_mar = 0
        self.t = datetime.now()
        self.frame = 0
        self.fatigue_prediction = 0

        self.fatigue_predictions_history = []
        self.MAX_PREDICTIONS_HISTORY = 10

    def __eye_aspect_ratio(self, eye):
        """
        Eye Aspect Ratio
        """
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    def __mouth_aspect_ratio(self, mouth):
        """
        Mouth Aspect Ratio
        """
        mouth_width = dist.euclidean(mouth[0], mouth[6])
        mouth_height = dist.euclidean(mouth[14], mouth[18])

        mar = mouth_height / mouth_width

        return mar

    def get_avg_ear(self) -> float:
        """
        Get Average Eye Aspect Ratio
        """
        return self.avg_ear

    def get_avg_mar(self) -> float:
        """
        Get Average Mouth Aspect Ratio
        """
        return self.avg_mar

    def run(self, landmarks) -> None:
        """
        Run Fatigue Detection System
        """

        # Calculate time difference between frames
        t1 = datetime.now()
        dt = t1 - self.t
        no_frame = self.frame + 1

        ear = self.eyes_features_extraction(landmarks)
        mar = self.mouth_features_extraction(landmarks)
        fatigue_prediction = self.fatigue_predictor_model(t1, dt, no_frame, ear, mar)

        # Save fatigue prediction to history
        self.update_history(t1, dt, no_frame, ear, mar, fatigue_prediction)

        return fatigue_prediction

    def update_history(self, t1, dt, no_frame, ear, mar, fatigue_prediction) -> None:
        """
        Update Fatigue Predictions History
        """

        # Save fatigue prediction to history
        self.fatigue_predictions_history.append(
            (
                t1,
                dt.total_seconds(),
                no_frame,
                ear,
                mar,
                fatigue_prediction,
            )
        )

        # Remove oldest fatigue prediction from history
        if len(self.fatigue_predictions_history) > self.MAX_PREDICTIONS_HISTORY:
            self.fatigue_predictions_history.pop(0)

        # Update values
        self.t = t1
        self.frame = no_frame

    def eyes_features_extraction(self, landmarks) -> float:
        """
        Closed Eyes Model
        """

        self.avg_ear = 0

        for feature in landmarks:
            left_eye = feature[42:48]
            right_eye = feature[36:42]

            left_ear = self.__eye_aspect_ratio(left_eye)
            right_ear = self.__eye_aspect_ratio(right_eye)

            self.avg_ear = (left_ear + right_ear) / 2.0

        return self.avg_ear

    def mouth_features_extraction(self, landmarks) -> float:
        """
        Open Mouth Model
        """

        self.avg_mar = 0

        for feature in landmarks:
            mouth = feature[48:68]

            mar = self.__mouth_aspect_ratio(mouth)

            self.avg_mar = mar

        return self.avg_mar

    def fatigue_predictor_model(self, t1, dt, no_frame, ear, mar) -> float:
        """Fatigue Predictor Model. It returns a value between 0 and 1.
        It uses a machine learning model to predict fatigue based on:
        - Time difference between frames
        - Number of frames
        - Eye Aspect Ratio
        - Mouth Aspect Ratio
        - The history of fatigue predictions
        """

        # Define some rules to start predicting fatigue

        # If the person is blinking, don't predict fatigue
        # If the person is talking, don't predict fatigue
        # If the person is not blinking, don't predict fatigue
        # If the person is not talking, don't predict fatigue
        # If the person is not blinking and not talking, predict fatigue
        # If the person open mouth for more than 1 second, predict fatigue, otherwise don't predict fatigue
        # If the person closed eyes for more than 1 second, predict fatigue, otherwise don't predict fatigue

        return 0