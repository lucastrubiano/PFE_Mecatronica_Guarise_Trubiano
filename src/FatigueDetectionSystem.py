from scipy.spatial import distance as dist
from datetime import datetime
import numpy as np
from config import(
     EAR_THRESHOLD,
     MAR_THRESHOLD,
     CONSECUTIVE_SEC_PERCLOS_THRESHOLD,
     CONSECUTIVE_SEC_POM_THRESHOLD,
     CONSECUTIVE_SEC_POY_THRESHOLD,
     PERCLOS_THRESHOLD,
     POY_THRESHOLD,
     YAWN_THRESHOLD,
     LEFT_EYE_LMS_NUMS,
     RIGHT_EYE_LMS_NUMS,
     CENTER_MOUTH_LMS_NUMS,
     BORDER_MOUTH_LMS_NUMS,
     FPS
    )

class FatigueDetectionSystem:
    def __init__(self) -> None:
        """
        Initialize Fatigue Detection System. This class contains all the methods needed to calculate the necessaries featurates to detect fatigue.
        """

        self.avg_ear = 0
        self.avg_mar = 0
        self.frame = 0
        self.fatigue_prediction = 0
        self.yawn_state = False
        self.last_yawn = 0
        self.ear_history = []
        self.mar_history = []
        self.yawn_history = []
        

    @staticmethod
    def _calc_ear_eye(vertical_left_points: list, vertical_right_points: list, horizontal_poitns: list) -> float:
        """
        Computer the EAR score for a single eyes given it's keypoints

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

    def _mouth_aspect_ratio(self, center_mouth_coord: list, border_mouth_coord: list) -> float:
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
    
    def eyes_features_extraction(self, landmarks:list, left_eye_lms_nums: list = LEFT_EYE_LMS_NUMS, right_eye_lms_nums: list = RIGHT_EYE_LMS_NUMS) -> float:
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
        if landmarks.shape[0]!=0:
            left_key_points = landmarks[left_eye_lms_nums, :2]
            right_key_points = landmarks[right_eye_lms_nums,:2]
            ear_left = self._calc_ear_eye(left_key_points[2:4], left_key_points[4:6], left_key_points[:2])  # computing the left eye EAR score
            ear_right = self._calc_ear_eye(right_key_points[2:4], right_key_points[4:6], right_key_points[:2])  # computing the right eye EAR score

            # computing the average EAR score
            return (ear_left + ear_right) / 2
        else:
            return 0
    
    def calc_perclos(self, ear: float, ear_history: list, ear_threshold: float = EAR_THRESHOLD, sec_perclos: float = CONSECUTIVE_SEC_PERCLOS_THRESHOLD, fps: float = FPS) -> (float, list):
        """
        Calculate the PERCLOS which is the eye's closure duration/ percentage of eye closure.
        It is to say the duration of the closure of the eyes in a given duration.

        Parameters
        ----------
            - ear (float): Eye Aspect Ratio
            - ear_history (list): List of eye states (0: open, 1: closed)
            - ear_threshold (float): Eye Aspect Ratio threshold
            - sec_perclos (float): PERCLOS duration in seconds
            - fps (float): Frames per second
        
        Returns
        --------
            - float: PERCLOS
            - list: List of eye states (0: open, 1: closed)
        """
        eye_state = 0
        perclos = 0
        if ear < ear_threshold:
            #If the ear is less than EAR_THRESHOLD then it is considered that the eye is closed
            eye_state = 1
        ear_history.append(eye_state)

        if self.frame >= sec_perclos * fps:
            # PERCLOS is the duration of the closure of the eyes. In our list of ear_history we storage the closure of the eyes
            # during a determinated amount of frames wich represents the durations of 1 minute
            perclos = np.mean(ear_history)
            ear_history.pop(0)

        return perclos, ear_history
    
    def calc_pom(self, mar:float, mar_history: list, mar_threshold: list = MAR_THRESHOLD, sec_pom: float = CONSECUTIVE_SEC_POM_THRESHOLD, fps: float = FPS) -> (float, list):
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
            #If the mar is more than MAR_THRESHOLD then it is considered that the mouth is opened
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
    
    def is_yawn(self, pom: float) -> bool:
        """
        Update yawn history and return True if it is a yawn

        Returns
        --------
            - bool: If it is a yawn, True is returned
        """
        if pom > YAWN_THRESHOLD:
            return True
        else:
            return False

    def calc_poy(self, yawn_state: float, last_yawn: float, yawn_history: list)-> (float, list):
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
            yawn_history.append((yawn_state,self.frame))

        if len(yawn_history)>0:
            
            if (self.frame - yawn_history[-1][1]) >= CONSECUTIVE_SEC_POY_THRESHOLD * FPS:
                yawn_history.pop(0)

        poy = sum([i[0] for i in yawn_history])

        return poy, yawn_history

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

    def fatigue_predictor_model(self, perclos: float, poy: float) -> int:
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

        # Define some rules to start predicting fatigue
        if perclos> PERCLOS_THRESHOLD:
            return 1
        
        if  poy>= POY_THRESHOLD:
            return 1
        
        return 0

    def run(self, landmarks: list) -> None:
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
        self.perclos, self.ear_history = self.calc_perclos(self.avg_ear, self.ear_history)
        self.pom, self.mar_history = self.calc_pom(self.avg_mar, self.mar_history)
        self.yawn_state = self.is_yawn(self.pom)
        self.poy, self.yawn_history = self.calc_poy(self.yawn_state, self.last_yawn, self.yawn_history)
        self.last_yawn = self.yawn_state
        fatigue_prediction = self.fatigue_predictor_model(self.perclos, self.poy)

        return fatigue_prediction