from scipy.spatial import distance as dist


from config import(
     EAR_THRESHOLD, 
     MAR_THRESHOLD, 
     CONSECUTIVE_SEC_PERCLOS_THRESHOLD,
     CONSECUTIVE_SEC_POM_THRESHOLD,
     CONSECUTIVE_SEC_POY_THRESHOLD,
     PERCLOS_THRESHOLD,
     EYES_LMS_NUMS,
     YAWN_THRESHOLD,
     LEFT_EYE_LMS_NUMS,
     RIGHT_EYE_LMS_NUMS,
     CENTER_MOUTH_LMS_NUMS,
     BORDER_MOUTH_LMS_NUMS,
     FPS
    )

from datetime import datetime


import numpy as np

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
        self.yawn_state = False
        self.last_yawn = 0

        self.fatigue_predictions_history = []
        self.ear_history = []
        self.mar_history = []
        self.yawn_history = []
        self.MAX_PREDICTIONS_HISTORY = 10


    @staticmethod
    def _calc_ear_eye(eye_pts):
        """
        Computer the EAR score for a single eyes given it's keypoints
        :param eye_pts: numpy array of shape (6,2) containing the keypoints of an eye
        :return: ear_eye
            EAR of the eye
        """
        A = dist.euclidean(eye_pts[2], eye_pts[3])
        B = dist.euclidean(eye_pts[4], eye_pts[5])
        C = dist.euclidean(eye_pts[0], eye_pts[1])

        ear = (A + B) / (2.0 * C)
        '''
        EAR is computed as the mean of two measures of eye opening (see mediapipe face keypoints for the eye)
        divided by the eye lenght
        '''
        return ear

    def _mouth_aspect_ratio(self, center_mouth_coord, border_mouth_coord):
        """
        Mouth Aspect Ratio
        """
        mouth_width = dist.euclidean(*border_mouth_coord)
        mouth_height = dist.euclidean(*center_mouth_coord)

        mar = mouth_height / mouth_width

        return mar
    
    def eyes_features_extraction(self,landmarks):
        """
        Computes the average eye aperture rate of the face

        Parameters
        ----------
        landmarks: landmarks: numpy array
            List of 478 mediapipe keypoints of the face

        Returns
        -------- 
        ear_score: float
            EAR average score between the two eyes
            The EAR or Eye Aspect Ratio is computed as the eye opennes divided by the eye lenght
            Each eye has his scores and the two scores are averaged
        """
        if landmarks.shape[0]!=0:
            ear_left = self._calc_ear_eye(landmarks[LEFT_EYE_LMS_NUMS, :2])  # computing the left eye EAR score
            ear_right = self._calc_ear_eye(landmarks[RIGHT_EYE_LMS_NUMS, :2])  # computing the right eye EAR score

            # computing the average EAR score
            return (ear_left + ear_right) / 2
        else:
            return 0
    
    def calc_perclos(self, ear: float) -> float:
        """
        Calculate the PERCLOS which is the eye's closure duration/ percentage of eye closure.
        It is to say the duration of the closure of the eyes in a given duration.

        Args:
            ear (float): Eye Aspect Ratio
        """
        eye_state = 0
        perclos = 0
        if ear < EAR_THRESHOLD:
            #If the ear is less than EAR_THRESHOLD then it is considered that the eye is closed
            eye_state = 1
        self.ear_history.append(eye_state)

        if self.frame >= CONSECUTIVE_SEC_PERCLOS_THRESHOLD * FPS:
            # PERCLOS is the duration of the closure of the eyes. In our list of ear_history we storage the closure of the eyes
            # during a determinated amount of frames wich represents the durations of 1 minute
            perclos = np.mean(self.ear_history)
            self.ear_history.pop(0)

        return perclos
    
    def calc_pom(self, mar:float) -> float:
        """
        Calculate the POM which is the mouth's opening duration/ percentage of mouth opening.

        Args:
            mar (float): Mouth Aspect Ratio
        
        Returns:
            float: POM
        """
        
        mouth_state = 0
        pom = 0
        if mar > MAR_THRESHOLD:
            #If the mar is more than MAR_THRESHOLD then it is considered that the mouth is opened
            mouth_state = 1
        self.mar_history.append(mouth_state)

        if self.frame >= CONSECUTIVE_SEC_POM_THRESHOLD * FPS:
            # POM is the duration of the opening of the mouth. In our list of mar_history we storage the opening of the mouth
            # during a determinated amount of frames wich represents the durations of 1 minute
            pom = np.mean(self.mar_history)
            self.mar_history.pop(0)

        return pom

    def get_perclos(self):
        """
        Get PERCLOS
        """
        return self.perclos
    
    def get_pom(self):
        """
        Get PERCLOS
        """
        return self.pom

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
    
    def get_poy(self) -> float:
        """
        Get POY

        Returns:
            float: POY
        """
        return self.poy
    
    def is_yawn(self, pom: float) -> bool:
        """
        Update yawn history and return True if it is a yawn

        Returns:
            bool: If it is a yawn, True is returned
        """
        if pom > YAWN_THRESHOLD:
            return True
        else:
            return False

    def calc_poy(self, yawn_state: float, last_yawn: float, yawn_history: list, frame: int)-> float:
        """
        Calculate the POY which is the number of yawn during a given time

        Args:
            yawn_state (float): The current yawn state
            last_yawn (float): The last yawn state

        Returns:
            float: POY
        """

        if yawn_state != last_yawn:
            yawn_history.append((yawn_state,frame))

        if len(yawn_history)>0:
            
            if (frame - yawn_history[-1][1]) >= CONSECUTIVE_SEC_POY_THRESHOLD * FPS:
                yawn_history.pop(0)

        poy = sum([i[0] for i in yawn_history])

        return poy, yawn_history


    def run(self, landmarks) -> None:
        """
        Run Fatigue Detection System
        """

        # Calculate time difference between frames
        t1 = datetime.now()
        dt = t1 - self.t
        self.frame += 1

        self.avg_ear = self.eyes_features_extraction(landmarks)
        self.avg_mar = self.mouth_features_extraction(landmarks)
        self.perclos = self.calc_perclos(self.avg_ear)
        self.pom = self.calc_pom(self.avg_mar)
        self.yawn_state = self.is_yawn(self.pom)
        self.poy, self.yawn_history = self.calc_poy(self.yawn_state, self.last_yawn, self.yawn_history, self.frame)
        self.last_yawn = self.yawn_state
        #fatigue_prediction = self.fatigue_predictor_model(t1, dt, self.frame, self.avg_ear, self.avg_mar)
        fatigue_prediction = 0
        # Save fatigue prediction to history
        self.update_history(t1, dt, self.frame, self.avg_ear, self.avg_mar, fatigue_prediction)

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

    # def eyes_features_extraction(self, landmarks) -> float:
    #     """
    #     Closed Eyes Model
    #     """

    #     self.avg_ear = 0

    #     for feature in landmarks:
    #         left_eye = feature[42:48]
    #         right_eye = feature[36:42]

    #         left_ear = self.__eye_aspect_ratio(left_eye)
    #         right_ear = self.__eye_aspect_ratio(right_eye)

    #         self.avg_ear = (left_ear + right_ear) / 2.0

    #     return self.avg_ear

    def mouth_features_extraction(self, landmarks) -> float:
        """
        Open Mouth Model
        """
        if landmarks.shape[0]:
            return self._mouth_aspect_ratio(landmarks[CENTER_MOUTH_LMS_NUMS], landmarks[BORDER_MOUTH_LMS_NUMS])
        else:
            return 0

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
        # * If the person open mouth for more than 1 second, predict fatigue, otherwise don't predict fatigue
        # * If the person closed eyes for more than 1 second, predict fatigue, otherwise don't predict fatigue
        return 1 if  self.calc_perclos(ear)> PERCLOS_THRESHOLD else 0