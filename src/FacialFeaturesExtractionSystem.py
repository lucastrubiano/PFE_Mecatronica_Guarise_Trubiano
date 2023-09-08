import cv2
import dlib
from imutils import face_utils
from config import LANDMARKS_MODEL


class FacialFeaturesExtractionSystem:
    def __init__(self) -> None:
        """
        Initialize dlib's face detector (HOG-based) and facial landmark predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(LANDMARKS_MODEL)

    def run(self, frame) -> list:
        """
        Run Facial Features Extraction Model
        """

        processed_frame = self.pre_processing(frame)

        faces_vector = self.face_detection(processed_frame)

        landmarks = self.landmarks_predictor(processed_frame, faces_vector)

        return landmarks

    def pre_processing(self, frame) -> None:
        """
        Pre-processing frame
        """

        # maybe resize frame?

        # convert frame to grayscale
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return processed_frame

    def face_detection(self, frame) -> list:
        """
        Detect faces in frame
        """

        # detect faces in the grayscale frame
        faces_vector = self.detector(frame)

        # choose the first face
        if faces_vector:
            face = faces_vector[0]
        else:
            return []

        # other possible options:
        # choose the biggest face
        # choose the closest face
        # choose the face with the highest confidence
        # choose the most centered face

        return [face]

    def landmarks_predictor(self, frame, faces_vector) -> list:
        """
        Predict landmarks in frame
        """

        # initialize landmarks vector
        landmarks = []

        # loop over the face detections
        for face in faces_vector:
            # determine the facial landmarks for the face region
            shape = self.predictor(frame, face)

            # convert facial landmark (x, y)-coordinates to a NumPy array
            shape = face_utils.shape_to_np(shape)

            # append landmarks to vector
            landmarks.append(shape)

        return landmarks
