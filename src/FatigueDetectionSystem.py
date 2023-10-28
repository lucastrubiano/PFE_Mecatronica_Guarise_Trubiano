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
from config import FPS
from config import LEFT_EYE_LMS_NUMS
from config import RIGHT_EYE_LMS_NUMS
from config import STATE_DISTRACTED
from config import STATE_FATIGUE
from config import STATE_NORMAL
from config import THRESHOLD_EAR
from config import THRESHOLD_MAR
from config import THRESHOLD_PERCLOS
from config import THRESHOLD_PERCLOS_PITCH
from config import THRESHOLD_PERCLOS_POY
from config import THRESHOLD_PITCH
from config import THRESHOLD_POM
from config import THRESHOLD_POY
from config import THRESHOLD_YAW
from config import THRESHOLD_YAWPER


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
        self.yamper = 0
        self.ear_history = 0
        self.yaw_history = 0
        self.mar_history = 0
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

    def calc_perclos(
        self, ear: float, ear_history: int, yaw: float, yaw_threshold: float = THRESHOLD_YAW,
        ear_threshold: float = THRESHOLD_EAR, sec_perclos: float = CONSECUTIVE_SEC_PERCLOS_THRESHOLD, fps: float = FPS,
    ) -> (float, list):
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

        if ear < ear_threshold and ear_history < (sec_perclos * fps):
            # If the ear is less than EAR_THRESHOLD then it is considered that the eye is closed
            ear_history += 1
        else:
            ear_history = ear_history - 1 if ear_history > 0 else 0

        perclos = ear_history / (sec_perclos * fps)

        perclos = perclos if perclos < 1 else 1

        return perclos, ear_history

    def calc_yawper(
        self, yaw: float, yaw_history: int, yaw_threshold: float = THRESHOLD_YAW,
        sec_yamper: float = CONSECUTIVE_SEC_YAW_THRESHOLD, fps: float = FPS,
    ) -> (float, int):
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

        if abs(yaw) > yaw_threshold and yaw_history < (sec_yamper * fps):
            yaw_history += 1
        else:
            yaw_history = yaw_history - 1 if yaw_history > 0 else 0

        yamper = yaw_history / (sec_yamper * fps)

        yamper = yamper if yamper < 1 else 1

        return yamper, yaw_history

    def calc_pom(
        self, mar: float, mar_history: int, mar_threshold: float = THRESHOLD_MAR,
        sec_pom: float = CONSECUTIVE_SEC_POM_THRESHOLD, fps: float = FPS,
    ) -> (float, int):
        """
        Calculate the POM which is the mouth's opening duration/ percentage of mouth opening.

        Parameters
        ----------
            - mar (float): Mouth Aspect Ratio
            - mar_history (int): Mouth history
            - mar_threshold (float): Mouth Aspect Ratio threshold
            - sec_pom (float): POM duration in seconds
            - fps (float): Frames per second

        Returns
        --------
            - float: POM
            - int: Mouth history
        """

        if mar > mar_threshold:
            # If the mar is more than MAR_THRESHOLD then it is considered that the mouth is opened
            mar_history += 1
        else:
            mar_history = mar_history - 1 if mar_history > 0 else 0

        pom = mar_history / (sec_pom * fps)

        pom = pom if pom < 1 else 1

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
        if pom > THRESHOLD_POM:
            return True
        else:
            return False

    def calc_poy(
        self, yawn_state: float, last_yawn: float, yawn_history: list,
        sec_pom: float = CONSECUTIVE_SEC_POY_THRESHOLD, fps: float = FPS,
    ) -> (float, int):
        """
        Calculate the POY which is the number of yawn during a given time

        Parameters
        ----------
            - yawn_state (float): The current yawn state
            - last_yawn (float): The last yawn state
            - yawn_history (list): List of yawn states (0: not yawn, 1: yawn)
            - sec_pom (float): POY duration in seconds
            - fps (float): Frames per second

        Returns
        --------
            - float: POY
            - list: List of yawn states (0: not yawn, 1: yawn)
        """

        if yawn_state != last_yawn:
            yawn_history.append((yawn_state, self.frame))

        if len(yawn_history) > 0:

            if (self.frame - yawn_history[0][1]) >= (sec_pom * fps):
                yawn_history.pop(0)

        poy = sum([i[0] for i in yawn_history])

        last_yawn = yawn_state
        return poy, yawn_history, last_yawn

    def mouth_features_extraction(
        self, landmarks: list,
        center_mouth: list = CENTER_MOUTH_LMS_NUMS, border_mouth: list = BORDER_MOUTH_LMS_NUMS,
    ) -> float:
        """
        Open Mouth Model

        Parameters
        ----------
            - landmarks (list): List of landmarks
            - center_mouth (list): Center mouth landmarks
            - border_mouth (list): Border mouth landmarks

        Returns
        --------
            - float: Mouth Aspect Ratio
        """
        if landmarks.shape[0] != 0:
            return self._mouth_aspect_ratio(landmarks[center_mouth], landmarks[border_mouth])
        else:
            return 0

    def state_predictor_model(
            self, perclos: float, poy: float, pitch: float, yawper: float,
            threshold_poy: int = THRESHOLD_POY, threshold_perclos_poy: float = THRESHOLD_PERCLOS_POY,
            threshold_picth: int = THRESHOLD_PITCH, threshold_perclos_pitch: float = THRESHOLD_PERCLOS_PITCH, threshold_yamper: float = THRESHOLD_YAWPER, ) -> str:
        """
        Fatigue Predictor Model

        Parameters
        ----------
            - perclos (float): PERCLOS
            - poy (float): POY
            - pitch (float): Pitch
            - yawper (float): YAMPER
            - threshold_poy (int): POY threshold
            - threshold_perclos_poy (float): PERCLOS threshold after pass POY threshold
            - threshold_picth (int): Pitch threshold
            - threshold_perclos_pitch (float): PERCLOS threshold after pass Pitch threshold
            - threshold_yamper (float): YAMPER threshold

        Returns
        --------
            - str: Fatigue prediction
        """
        pitch = pitch if pitch else 0

        if poy >= threshold_poy:
            perclos_threshold = threshold_perclos_poy

        elif abs(pitch) >= threshold_picth:
            perclos_threshold = threshold_perclos_pitch
        else:
            perclos_threshold = THRESHOLD_PERCLOS

        if perclos > perclos_threshold:
            return STATE_FATIGUE
        elif yawper > threshold_yamper:
            return STATE_DISTRACTED

        return STATE_NORMAL

    def run(self, landmarks: list, pitch: float, yaw: float) -> str:
        """
        Run Fatigue Detection System. Return the state prediction

        Parameters
        ----------
            - landmarks (list): List of landmarks
            - pitch (float): Pitch
            - yaw (float): Yaw

        Returns
        --------
            - str: State prediction
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
        if yaw:
            self.yawper, self.yaw_history = self.calc_yawper(
                yaw, self.yaw_history,
            )
        state_prediction = self.state_predictor_model(
            self.perclos, self.poy, pitch, self.yawper,
        )

        return state_prediction
