from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.spatial import distance as dist

from config import BORDER_MOUTH_LMS_NUMS
from config import CENTER_MOUTH_LMS_NUMS
from config import CONSECUTIVE_SEC_PERCLOS_THRESHOLD
from config import CONSECUTIVE_SEC_POM_THRESHOLD
from config import CONSECUTIVE_SEC_POY_THRESHOLD
from config import CONSECUTIVE_SEC_YAW_THRESHOLD
from config import DISTRACTED_STATE
from config import EAR_THRESHOLD
from config import FATIGUE_STATE
from config import FPS
from config import LEFT_EYE_LMS_NUMS
from config import MAR_THRESHOLD
from config import NORMAL_STATE
from config import RIGHT_EYE_LMS_NUMS
from config import THRESHOLD_PERCLOS
from config import THRESHOLD_PERCLOS_PITCH
from config import THRESHOLD_PERCLOS_POY
from config import THRESHOLD_POY
from config import THRESHOLD_YAW
from config import THRESHOLD_YAWN


class FatigueDetectionSystem:
    """
    This class contains all the methods needed to calculate the necessaries featurates to detect fatigue.
    """

    def __init__(self) -> None:
        """
        Initialize Fatigue Detection System.
        """

        self.avg_ear = 0
        self.avg_mar = 0
        self.frame = 0
        self.fatigue_prediction = 0
        self.yawn_state = False
        self.last_yawn = 0
        self.ear_history = 0
        self.yaw_history = 0
        self.mar_history = []
        self.yawn_history = []

    @staticmethod
    def _calc_ear_eye(vertical_left_points: list, vertical_right_points: list, horizontal_poitns: list) -> float:
        """
        Compute the EAR score for a single eyes given it's keypoints

        Parameters
        ----------
            - vertical_left_points (list): Vertical left points
            - vertical_right_points (list): Vertical right points
            - horizontal_poitns (list): Horizontal points

        Returns
        --------
            - float: EAR score
        """
        A = dist.euclidean(*vertical_right_points)
        B = dist.euclidean(*vertical_left_points)
        C = dist.euclidean(*horizontal_poitns)

        ear = (A + B) / (2.0 * C)

        return ear

    @staticmethod
    def _mouth_aspect_ratio(center_mouth_coord: list, border_mouth_coord: list) -> float:
        """
        Mouth Aspect Ratio

        Parameters
        ----------
            - center_mouth_coord (list): Center mouth coordinates
            - border_mouth_coord (list): Border mouth coordinates

        Returns
        --------
            - float: MAR score
        """
        mouth_width = dist.euclidean(*border_mouth_coord)
        mouth_height = dist.euclidean(*center_mouth_coord)

        mar = mouth_height / mouth_width

        return mar

    def eyes_features_extraction(self, landmarks: list, left_eye_lms_nums: list = LEFT_EYE_LMS_NUMS, right_eye_lms_nums: list = RIGHT_EYE_LMS_NUMS) -> float:
        """
        Computes the average eye aperture rate of the face

        Parameters
        ----------
            - landmarks (list): List of 478 mediapipe keypoints of the face
            - left_eye_lms_nums (list): List of the left eye keypoints
            - right_eye_lms_nums (list): List of the right eye keypoints

        Returns
        --------
            - ear_score (float): EAR average score between the two eyes.
            The EAR or Eye Aspect Ratio is computed as the eye opennes divided by the eye lenght.
            Each eye has his scores and the two scores are averaged
        """
        if landmarks.shape[0] != 0:
            left_key_points = landmarks[left_eye_lms_nums, :2]
            right_key_points = landmarks[right_eye_lms_nums, :2]
            # computing the left eye EAR score
            ear_left = self._calc_ear_eye(
                left_key_points[2:4], left_key_points[4:6], left_key_points[:2],
            )
            # computing the right eye EAR score
            ear_right = self._calc_ear_eye(
                right_key_points[2:4], right_key_points[4:6], right_key_points[:2],
            )

            # computing the average EAR score
            return (ear_left + ear_right) / 2
        else:
            return 0

    def calc_perclos(self, ear: float, ear_history: int, yaw: float, yaw_threshold: float = THRESHOLD_YAW, ear_threshold: float = EAR_THRESHOLD, sec_perclos: float = CONSECUTIVE_SEC_PERCLOS_THRESHOLD, fps: float = FPS) -> (float, list):
        """
        Calculate the `PERCLOS` which is the eye's closure duration/ percentage of eye closure.
        It is to say the duration of the closure of the eyes in a given duration.

        Parameters
        ----------
            - ear (float): Eye Aspect Ratio
            - ear_history (int): History of eyes states
            - yaw (float): Yaw
            - yaw_threshold (float): Yaw threshold
            - ear_threshold (float): Eye Aspect Ratio threshold
            - sec_perclos (float): PERCLOS duration in seconds
            - fps (float): Frames per second

        Returns
        --------
            - float: PERCLOS
            - int: History of eyes states
        """

        if ear < ear_threshold and abs(yaw) < yaw_threshold:
            # If the ear is less than EAR_THRESHOLD then it is considered that the eye is closed
            ear_history += 1
        else:
            ear_history = ear_history - 1 if ear_history > 0 else 0

        perclos = ear_history / (sec_perclos * fps)

        return perclos, ear_history

    def calc_yawper(self, yaw: float, yaw_history: int, yaw_threshold: float = THRESHOLD_YAW, sec_yamper: float = CONSECUTIVE_SEC_YAW_THRESHOLD, fps: float = FPS) -> (float, int):
        """
        Calculate the `YAWPER` which is the yaw more than a given threshold duration/ percentage of yaw.

        Parameters
        ----------
            - yaw (float): Yaw
            - yaw_history (int): History of yaw states
            - yaw_threshold (float): Yaw threshold
            - sec_yamper (float): YAMPER duration in seconds
            - fps (float): Frames per second

        Returns
        --------
            - float: YAMPER
            - int: History of yaw states
        """

        if abs(yaw) > yaw_threshold:
            yaw_history += 1
        else:
            yaw_history = yaw_history - 1 if yaw_history > 0 else 0

        yamper = yaw_history / (sec_yamper * fps)

        return yamper, yaw_history

    def calc_pom(self, mar: float, mar_history: list, mar_threshold: list = MAR_THRESHOLD, sec_pom: float = CONSECUTIVE_SEC_POM_THRESHOLD, fps: float = FPS) -> (float, list):
        """
        Calculate the POM which is the mouth's opening duration/ percentage of mouth opening.

        Parameters
        ----------
            - mar (float): Mouth Aspect Ratio
            - mar_history (list): List of mouth states (0: closed, 1: open)
            - mar_threshold (float): Mouth Aspect Ratio threshold
            - sec_pom (float): POM duration in seconds
            - fps (float): Frames per second

        Returns
        --------
            - float: POM
            - list: List of mouth states (0: closed, 1: open)
        """

        mouth_state = 0
        pom = 0
        if mar > mar_threshold:
            # If the mar is more than MAR_THRESHOLD then it is considered that the mouth is opened
            mouth_state = 1
        mar_history.append(mouth_state)

        if self.frame >= sec_pom * fps:
            # POM is the duration of the opening of the mouth. In our list of mar_history we storage the opening of the mouth
            # during a determinated amount of frames wich represents the durations of 1 minute
            pom = np.mean(mar_history)
            mar_history.pop(0)

        return pom, mar_history

    def get_perclos(self) -> float:
        """
        Get PERCLOS

        Returns
        --------
            - float: PERCLOS
        """
        return self.perclos

    def get_pom(self) -> float:
        """
        Get PERCLOS

        Returns
        --------
            - float: PERCLOS
        """
        return self.pom

    def get_avg_ear(self) -> float:
        """
        Get Average Eye Aspect Ratio

        Returns
        --------
            - float: Average Eye Aspect Ratio
        """
        return self.avg_ear

    def get_avg_mar(self) -> float:
        """
        Get Average Mouth Aspect Ratio
        """
        return self.avg_mar

    def get_poy(self) -> float:
        """
        Get POY

        Returns
        --------
            - float: POY
        """
        return self.poy

    def get_yawper(self) -> float:
        """
        Get YAMPER

        Returns
        --------
            - float: YAMPER
        """
        return self.yawper

    @staticmethod
    def is_yawn(pom: float) -> bool:
        """
        Update yawn history and return True if it is a yawn

        Returns
        --------
            - bool: If it is a yawn, True is returned
        """
        if pom > THRESHOLD_YAWN:
            return True
        else:
            return False

    def calc_poy(self, yawn_state: float, last_yawn: float, yawn_history: list) -> (float, list):
        """
        Calculate the POY which is the number of yawn during a given time

        Parameters
        ----------
            - yawn_state (float): The current yawn state
            - last_yawn (float): The last yawn state
            - yawn_history (list): List of yawn states (0: not yawn, 1: yawn)

        Returns
        --------
            - float: POY
            - list: List of yawn states (0: not yawn, 1: yawn)
        """

        if yawn_state != last_yawn:
            yawn_history.append((yawn_state, self.frame))

        if len(yawn_history) > 0:

            if (self.frame - yawn_history[0][1]) >= CONSECUTIVE_SEC_POY_THRESHOLD * FPS:
                yawn_history.pop(0)

        poy = sum([i[0] for i in yawn_history])

        last_yawn = yawn_state
        return poy, yawn_history, last_yawn

    def mouth_features_extraction(self, landmarks: list) -> float:
        """
        Open Mouth Model

        Parameters
        ----------
            - landmarks (list): List of landmarks

        Returns
        --------
            - float: Mouth Aspect Ratio
        """
        if landmarks.shape[0]:
            return self._mouth_aspect_ratio(landmarks[CENTER_MOUTH_LMS_NUMS], landmarks[BORDER_MOUTH_LMS_NUMS])
        else:
            return 0

    def state_predictor_model(self, perclos: float, poy: float, pitch: float, yawper: float) -> str:
        """
        Fatigue Predictor Model

        Parameters
        ----------
            - perclos (float): PERCLOS
            - poy (float): POY

        Returns
        --------
            - int: Fatigue prediction
        """

        if poy >= THRESHOLD_POY:
            perclos_threshold = THRESHOLD_PERCLOS_POY

        elif abs(pitch) >= THRESHOLD_PERCLOS_PITCH:
            perclos_threshold = THRESHOLD_PERCLOS_PITCH
        else:
            perclos_threshold = THRESHOLD_PERCLOS

        if perclos > perclos_threshold:
            return FATIGUE_STATE
        elif yawper > THRESHOLD_YAW:
            return DISTRACTED_STATE

        return NORMAL_STATE

    def run(self, landmarks: list, pitch: float, yaw: float) -> None:
        """
        Run Fatigue Detection System

        Parameters
        ----------
            - landmarks (list): List of landmarks

        Returns
        --------
            - None
        """

        # Calculate time difference between frames
        self.frame += 1

        self.avg_ear = self.eyes_features_extraction(landmarks)
        self.avg_mar = self.mouth_features_extraction(landmarks)
        self.perclos, self.ear_history = self.calc_perclos(
            self.avg_ear, self.ear_history, yaw,
        )
        self.pom, self.mar_history = self.calc_pom(
            self.avg_mar, self.mar_history,
        )
        yawn_state = self.is_yawn(self.pom)
        self.poy, self.yawn_history, self.last_yawn = self.calc_poy(
            yawn_state, self.last_yawn, self.yawn_history,
        )
        self.yawper, self.yaw_history = self.calc_yawper(
            yaw, self.yaw_history,
        )
        fatigue_prediction = self.state_predictor_model(
            self.perclos, self.poy, pitch, yaw,
        )

        return fatigue_prediction
