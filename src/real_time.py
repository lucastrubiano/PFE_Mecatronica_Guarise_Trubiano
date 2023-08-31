import cv2
from os import environ as env

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


class RealTime:
    FACE_CASCADE_PATH = env["FACE_DEFAULT_MODEL"]
    EYE_CASCADE_PATH = env["EYE_DEFAULT_MODEL"]
    MOUTH_CASCADE_PATH = env["MOUTH_DEFAULT_MODEL"]
    VIDEO_CAPTURE_DEVICE = 0

    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(self.FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(self.EYE_CASCADE_PATH)
        self.mouth_cascade = cv2.CascadeClassifier(self.MOUTH_CASCADE_PATH)
        self.video_capture = cv2.VideoCapture(self.VIDEO_CAPTURE_DEVICE)

        self.RUNNING = False

    def run(self):
        self.RUNNING = True

        while self.RUNNING:
            ret, frame = self.video_capture.read()
            if frame.any():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                scale_factor_mouth = 3.5

                # Only process if there is a face detected and process the biggest one
                if len(faces) > 0:
                    face = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
                    x, y, w, h = face
                    # gray = gray[y : y + h, x : x + w]
                    # frame = frame[y : y + h, x : x + w]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE, 5)

                    # Draw a little circle at x, y
                    cv2.circle(frame, (x, y), 8, BLUE, -1)

                    # For eye detection take the upper 2/3 of the face rectangle
                    gray_eyes = gray[y : y + h * 2 // 3, x : x + w]
                    color_eyes = frame[y : y + h * 2 // 3, x : x + w]
                    cv2.rectangle(
                        frame, (x - 30, y), (x + w + 30, y + h * 2 // 3), BLUE, 2
                    )

                    # And for mouth detection take the lower half
                    gray_mouth = gray[y + h // 2 : y + h + 30, x : x + w]
                    color_mouth = frame[y + h // 2 : y + h + 30, x : x + w]
                    cv2.rectangle(
                        frame, (x - 50, y + h // 2), (x + w + 50, y + h + 30), BLUE, 2
                    )

                    eyes = self.eye_cascade.detectMultiScale(gray_eyes)
                    mouth = self.mouth_cascade.detectMultiScale(gray_mouth, scale_factor_mouth)

                    for ex, ey, ew, eh in eyes:
                        cv2.rectangle(
                            color_eyes, (ex, ey), (ex + ew, ey + eh), GREEN, 2
                        )

                    for mx, my, mw, mh in mouth:
                        cv2.rectangle(color_mouth, (mx, my), (mx + mw, my + mh), RED, 2)

            cv2.imshow("img", frame)

            # try:
            #     if len(mouth) > 1:
            #         bucle = True
            #         while bucle:
            #             eval(input("Enter command: "))
            # except:
            #     pass

            if cv2.waitKey(10) == ord("q"):  # wait until 'q' key is pressed
                self.RUNNING = False
                self.video_capture.release()
                break

            if cv2.waitKey(10) == ord("s"):
                scale_factor_mouth = float(input("Enter scale factor: "))

        cv2.destroyAllWindows()

    def stop(self):
        self.RUNNING = False
        self.video_capture.release()
        cv2.destroyAllWindows()
