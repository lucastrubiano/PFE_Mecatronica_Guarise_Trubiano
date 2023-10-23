import cv2
import mediapipe as mp
import numpy as np
from config import (
    EYES_LMS_NUMS,
    OUTTER_LIP_LMS_NUMS,
    INNER_LIP_LMS_NUMS,
    LEFT_IRIS_NUM,
    RIGHT_IRIS_NUM,
)

class FacialFeaturesExtractionSystem:
    """
    Class FacialFeaturesExtractionSystem is the class that runs the facial features extraction system.
    """

    def __init__(self) -> None:
        """
        Initialize Facial features extraction system which use mediapipe library
        """
        self.detector = mp.solutions.face_mesh.FaceMesh(
                                                static_image_mode=False,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5,
                                                refine_landmarks=True
                                                )

    def run(self, frame: np.array) -> (list, tuple):
        """
        Run Facial Features Extraction Model

        Parameters
        ----------
            - frame (np.array): Frame to be processed

        Returns
        --------
            - list: List of landmarks
            - frame_size: Size of the frame
        """

        processed_frame, frame_size = self.pre_processing(frame)

        landmarks = self.face_detection(processed_frame)
        
        return landmarks,frame_size
    
    def _get_main_face(self, multiple_face_landmakrs: np.array) -> np.array:
        """
        Get the main face from multiple faces detected

        Parameters
        ----------
            - multiple_face_landmakrs (np.array): Multiple Face landmarks

        Returns
        -------
            - np.array: Main Face landmarks
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

    def pre_processing(self, frame:np.array) -> (np.array, tuple):
        """
        Pre-processing frame. First the frame is transform in grayscale and then filter.

        Parameters
        ----------
            - frame (np.array): Frame to be processed

        Returns
        -------
            - np.array: Processed frame
            - tuple: frame size
        
        """

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
        Detect faces in frame and return the main face
        
        Parameters
        ----------
            - frame (np.array): Frame to be processed

        Returns
        -------
            - np.array: Main Face landmarks
        """

        # initialize landmarks vector
        mian_face = np.array([])
        # find the faces using the face mesh model
        multiple_face_landmakrs = self.detector.process(frame).multi_face_landmarks
        if multiple_face_landmakrs:  # process the frame only if at least a face is found
            # getting face landmarks and then take only the bounding box of the biggest face
            mian_face = self._get_main_face(multiple_face_landmakrs)

        return mian_face
    
    def show_eye_keypoints(self, color_frame: np.array, landmarks: list, frame_size: tuple, numerate_dots: bool = False, plot_iris: bool = False ,plot_eyes: bool = True, plot_inner_lips:bool = True, plot_outter_lips:bool = False) -> None :
        """
        Shows eyes keypoints found in the face, drawing red circles in their position in the frame/image

        Parameters
        ----------
            - color_frame (np.array): Frame/image in which the eyes keypoints are found
            - landmarks (list): List of landmarks
            - frame_size (tuple): Size of the frame
            - numerate_dots (bool): If True numerate the dots
            - plot_iris (bool): If True plot iris
            - plot_eyes (bool): If True plot eyes
            - plot_inner_lips (bool): If True plot inner lips
            - plot_outter_lips (bool): If True plot outter lips

        Returns
        -------
            - None
        """

        if landmarks.shape[0] != 0:
            if plot_iris:
                cv2.circle(color_frame, (landmarks[LEFT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
                    3, (255, 255, 255), cv2.FILLED)
                cv2.circle(color_frame, (landmarks[RIGHT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
                        3, (255, 255, 255), cv2.FILLED)
            dots_to_plot = []
            if plot_eyes:
                dots_to_plot.extend(EYES_LMS_NUMS)
            if plot_inner_lips:
                dots_to_plot.extend(INNER_LIP_LMS_NUMS)
            if plot_outter_lips:
                dots_to_plot.extend(OUTTER_LIP_LMS_NUMS)
            for n in dots_to_plot:
                x = int(landmarks[n, 0] * frame_size[0])
                y = int(landmarks[n, 1] * frame_size[1])
                cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
                if numerate_dots:
                    cv2.putText(
                        color_frame,
                        f'{n}',
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1,
                    )
