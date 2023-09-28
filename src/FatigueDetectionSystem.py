from scipy.spatial import distance as dist


from config import EAR_THRESHOLD, MAR_THRESHOLD, FRAMES_PER_SECONDS, PERCLOS_THRESHOLD, EYES_LMS_NUMS

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

        self.fatigue_predictions_history = []
        self.ear_history = []
        self.MAX_PREDICTIONS_HISTORY = 10


    @staticmethod
    def _calc_EAR_eye(eye_pts):
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

    def __mouth_aspect_ratio(self, mouth):
        """
        Mouth Aspect Ratio
        """
        mouth_width = dist.euclidean(mouth[0], mouth[6])
        mouth_height = dist.euclidean(mouth[14], mouth[18])

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

        # numpy array for storing the keypoints positions of the left and right eyes
        eye_pts_l = np.zeros(shape=(6, 2))
        eye_pts_r = eye_pts_l.copy()

        # get the face mesh keypoints
        for i in range(len(EYES_LMS_NUMS)//2):
            # array of x,y coordinates for the left eye reference point
            eye_pts_l[i] = landmarks[EYES_LMS_NUMS[i], :2]
            # array of x,y coordinates for the right eye reference point
            eye_pts_r[i] = landmarks[EYES_LMS_NUMS[i+6], :2]

        ear_left = self._calc_EAR_eye(eye_pts_l)  # computing the left eye EAR score
        ear_right = self._calc_EAR_eye(eye_pts_r)  # computing the right eye EAR score

        # computing the average EAR score
        ear_avg = (ear_left + ear_right) / 2

        self.avg_ear = ear_avg
        return ear_avg
    
    def __perclos(self, ear: float) -> float:
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

        if self.frame >= FRAMES_PER_SECONDS:
            # PERCLOS is the duration of the closure of the eyes. In our list of ear_history we storage the closure of the eyes
            # during a determinated amount of frames wich represents the durations of 1 minute
            perclos = np.mean(self.ear_history)
            self.ear_history.pop(0)
        
        return perclos



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
        self.frame += 1

        ear = self.eyes_features_extraction(landmarks)
        #mar = self.mouth_features_extraction(landmarks)
        mar = np.array([])
        fatigue_prediction = self.fatigue_predictor_model(t1, dt, self.frame, ear, mar)

        # Save fatigue prediction to history
        self.update_history(t1, dt, self.frame, ear, mar, fatigue_prediction)

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
        # * If the person open mouth for more than 1 second, predict fatigue, otherwise don't predict fatigue
        # * If the person closed eyes for more than 1 second, predict fatigue, otherwise don't predict fatigue
        return 1 if  self.__perclos(ear)> PERCLOS_THRESHOLD else 0