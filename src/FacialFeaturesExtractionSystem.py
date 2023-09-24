import cv2
import mediapipe as mp
import numpy as np
from config import (
    EYES_LMS_NUMS,
    LEFT_IRIS_NUM,
    RIGHT_IRIS_NUM
)

class FacialFeaturesExtractionSystem:
    def __init__(self) -> None:
        """
        Initialize mediapipe's face detector and facial landmark predictor
        """
        self.detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5,
                                               refine_landmarks=True)

    def run(self, frame) -> list:
        """
        Run Facial Features Extraction Model
        """

        processed_frame, frame_size = self.pre_processing(frame)

        landmarks = self.face_detection(processed_frame)

        return landmarks,frame_size
    
    def _get_main_face(self, multiple_face_landmakrs: np.array) -> np.array:
        """
        Get the main face from multiple faces detected

        Args:
            multiple_face_landmakrs (np.array): Multiple Face landmarks

        Returns:
            np.array: Main Face landmarks
        """
        surface = 0
        for face_lamdmarks in multiple_face_landmakrs:
            landmarks = [np.array([point.x, point.y, point.z]) \
                            for point in face_lamdmarks.landmark]

            landmarks = np.array(landmarks)

            dx = landmarks[:, 0].max() - landmarks[:, 0].min()
            dy = landmarks[:, 1].max() - landmarks[:, 1].min()
            new_surface = dx * dy
            if new_surface > surface:
                main_face = landmarks
        
        return main_face

    def pre_processing(self, frame) -> None:
        """
        Pre-processing frame
        """


        # start the tick counter for computing the processing time for each frame
        e1 = cv2.getTickCount()

        # transform the BGR frame in grayscale
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get the frame size
        frame_size = frame.shape[1], frame.shape[0]

        # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
        processed_frame = np.expand_dims(cv2.bilateralFilter(processed_frame, 5, 10, 10), axis=2)
        processed_frame = np.concatenate([processed_frame, processed_frame, processed_frame], axis=2)

        return processed_frame, frame_size

    def face_detection(self, frame: np.array) -> np.array:
        """
        Detect faces in frame
        """
        # initialize landmarks vector
        mian_face = np.array
        # find the faces using the face mesh model
        multiple_face_landmakrs = self.detector.process(frame).multi_face_landmarks
        if multiple_face_landmakrs:  # process the frame only if at least a face is found
            # getting face landmarks and then take only the bounding box of the biggest face
            mian_face = self._get_main_face(multiple_face_landmakrs)

        return mian_face
    
    def show_eye_keypoints(self, color_frame, landmarks, frame_size):
        """
        Shows eyes keypoints found in the face, drawing red circles in their position in the frame/image

        Parameters
        ----------
        color_frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: landmarks: numpy array
            List of 478 mediapipe keypoints of the face
        """

        self.keypoints = landmarks

        cv2.circle(color_frame, (landmarks[LEFT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
                   3, (255, 255, 255), cv2.FILLED)
        cv2.circle(color_frame, (landmarks[RIGHT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
                   3, (255, 255, 255), cv2.FILLED)

        for n in EYES_LMS_NUMS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return
