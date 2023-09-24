import cv2

from config import DIR_LOGS, LOG_FILE
from .FacialFeaturesExtractionSystem import FacialFeaturesExtractionSystem
from .FatigueDetectionSystem import FatigueDetectionSystem
from .AlertSystem import AlertSystem
from os import makedirs

from datetime import datetime
import os

# Save ear and mar values for each frame, to make some alalysis
log_file = "./logs/{}.csv"
SAVE_LOGS = False
CATEGORY = 'train'
PARAMS_TO_LOG = "log_all"
# bostezo, ojos_cerrados, ojos_abiertos, boca_abierta, hablando, bostezando

class RealTime:

    def __init__(self, camera: int = 0):
        """
        Initialize Real Time System

        Args:
            camera (int, optional): Camera input. Defaults to 0.
        """
        makedirs(DIR_LOGS, exist_ok=True)
        self.facial_features_extraction = FacialFeaturesExtractionSystem()
        self.fatigue_detection_system = FatigueDetectionSystem()
        self.alert_system = AlertSystem()
        self.camera = camera

    def run(self):

        # Start video capture
        video_frame = cv2.VideoCapture(self.camera)

        # Set cv2 optimized to true, if it is not
        if not cv2.useOptimized():
            try:
                cv2.setUseOptimized(True)
            except:
                print(
                    "OpenCV optimization could not be set to True, the script may be slower than expected")

        
        

        # Run the 3 subsystems in cascade
        while True:
            ret, frame = video_frame.read()
            if not ret:
                break

            # if the frame comes from webcam, flip it so it looks like a mirror.
            if self.camera == 0:
                frame = cv2.flip(frame, 2)

            landmarks,frame_size = self.facial_features_extraction.run(frame)

            self.facial_features_extraction.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size)
            print(landmarks)

            # fatigue_prediction = self.fatigue_detection_system.run(landmarks)
            # avg_ear = self.fatigue_detection_system.get_avg_ear()
            # avg_mar = self.fatigue_detection_system.get_avg_mar()

            # cv2.putText(
            #     frame,
            #     "EAR: {:.2f}".format(avg_ear),
            #     (300, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (0, 255, 0),
            #     2,
            # )
            # cv2.putText(
            #     frame,
            #     "MAR: {:.2f}".format(avg_mar),
            #     (300, 60),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (0, 255, 0),
            #     2,
            # )

            # if SAVE_LOGS and avg_ear != 0 and avg_mar != 0:

            #     # Press "space" to save fatigue detected frame
            #     if cv2.waitKey(1) & 0xFF == ord(" "):
            #         fatigue_prediction = 1
            #     else:
            #         fatigue_prediction = 0
                
            #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            #     # COLS: timestamp;path_img;ear;mar
            #     path_img = f"./frames/{CATEGORY}/"

            #     # mkdir
            #     if not os.path.exists(path_img):
            #         os.makedirs(path_img)

            #     full_path_img = "{}frame_{}.jpg".format(path_img, timestamp)

            #     # Save frame to disk
            #     cv2.imwrite(
            #         full_path_img,
            #         frame,
            #     )

            #     # Save ear and mar values for each frame, to make some alalysis
            #     with open(log_file.format(PARAMS_TO_LOG), "a+") as f:

            #         row_to_write = ";".join(map(str,[timestamp, full_path_img, avg_ear, avg_mar,list(*landmarks), fatigue_prediction])) 
            #         f.write(
            #            row_to_write + "\n"
            #         )

            # cv2.putText(
            #     frame,
            #     "Fatigue: {:.2f}".format(fatigue_prediction),
            #     (10, 60),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (0, 0, 255),
            #     2,
            # )

            # alert_result = self.alert_system.run(fatigue_prediction)


            # Show frame
            cv2.imshow("Fatigue Detector", frame)

            # Press "q" to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release video capture and close all windows when exiting
        cv2.destroyAllWindows()
        video_frame.release()


if __name__ == "__main__":
    RealTime.run()