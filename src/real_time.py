import cv2
from os import environ as env

class RealTime:
    FACE_CASCADE_PATH = env["FACE_DEFAULT_MODEL"]
    EYE_CASCADE_PATH = env["EYE_DEFAULT_MODEL"]
    VIDEO_CAPTURE_DEVICE = 0

    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(self.FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(self.EYE_CASCADE_PATH)
        self.video_capture = cv2.VideoCapture(self.VIDEO_CAPTURE_DEVICE)

        self.RUNNING = False

    def run(self):
        self.RUNNING = True

        while self.RUNNING:
            ret, frame = self.video_capture.read()
            if frame.any():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for x, y, w, h in faces:
                    img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y : y + h, x : x + w]
                    roi_color = img[y : y + h, x : x + w]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    for ex, ey, ew, eh in eyes:
                        cv2.rectangle(
                            roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                        )
            cv2.imshow("img", frame)

            if cv2.waitKey(10) == ord("q"):  # wait until 'q' key is pressed
                self.RUNNING = False
                self.video_capture.release()
                break

        cv2.destroyAllWindows()

    def stop(self):
        self.RUNNING = False
        self.video_capture.release()
        cv2.destroyAllWindows()
