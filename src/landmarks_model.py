import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
from config import (
    EAR_THRESHOLD,
    CONSECUTIVE_FRAMES_THRESHOLD,
    LANDMARKS_MODEL
)


class LandmarksModel():
    def __init__(self):
        # Initialize dlib's face detector (HOG-based) and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(LANDMARKS_MODEL)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def run(self):

        # Initialize counters
        consecutive_frames = 0

        # Start video capture
        self.video_frame = cv2.VideoCapture(0)

        while True:
            ret, frame = self.video_frame.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                shape = self.predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[42:48]
                right_eye = shape[36:42]

                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)

                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    consecutive_frames += 1
                    if consecutive_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                        cv2.putText(frame, "Fatigue Detected", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    consecutive_frames = 0
                #Write shape dots on frame
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, "EAR: {:.2f}".format(
                    avg_ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Fatigue Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def stop(self):
        self.video_frame.release()
        cv2.destroyAllWindows()
