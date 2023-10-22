import cv2

from config import DIR_LOGS, LOG_FILE, CONSECUTIVE_FRAMES_FPS, FILE_FPS
from .FacialFeaturesExtractionSystem import FacialFeaturesExtractionSystem
from .FatigueDetectionSystem import FatigueDetectionSystem
from .AlertSystem import AlertSystem
from os import makedirs
from math import floor

from datetime import datetime
import os, time

CATEGORY = 'train'
PARAMS_TO_LOG = "log_all"

class RealTime:

    def __init__(self, camera: int = 0, print_landmarks: bool = True, save_logs: bool = False, log_file: str = LOG_FILE) -> None:
        """
        Initialize Real Time System

        Parameters
        ----------

            - camera (int, optional): Camera input. Defaults to 0.
            - print_landmarks (bool, optional): Print the landmarks. Defaults to True.
            - save_logs (bool, optional): Save logs. Defaults to False.
            - log_file (str, optional): Log file. Defaults to "./logs/{}.csv".

        Returns
        -------

            - None
        """
        makedirs(DIR_LOGS, exist_ok=True)
        self.facial_features_extraction = FacialFeaturesExtractionSystem()
        self.fatigue_detection_system = FatigueDetectionSystem()
        self.alert_system = AlertSystem()
        self.camera = camera
        self.t0 = time.perf_counter()
        self.print_landmarks = print_landmarks
        self.save_logs = save_logs
        self.log_file = log_file
    
    def set_fps(self, consecutive_frames_fps: float = CONSECUTIVE_FRAMES_FPS, file_fps: float = FILE_FPS) -> None:

        """
        Set FPS of the device

        Parameters
        ----------

            - consecutive_frames_fps (float, optional): Consecutive frames. Defaults to CONSECUTIVE_FRAMES_FPS.
            - file_fps (float, optional): File FPS. Defaults to FILE_FPS.

        Returns
        -------

            - None
        """

        video_frame = cv2.VideoCapture(self.camera)

        # Set cv2 optimized to true, if it is not
        if not cv2.useOptimized():
            try:
                cv2.setUseOptimized(True)
            except:
                print(
                    "OpenCV optimization could not be set to True, the script may be slower than expected")
        
        counter = 0
        # Run the 3 subsystems in cascade
        while True:

            # If there is not ret
            ret, _ = video_frame.read()
            if not ret:
                break

            #Calculate FPS
            t_now = time.perf_counter()
            fps = self.calc_fps(t_now, self.t0, counter)

            if self.counter > consecutive_frames_fps:
                with open(file_fps, 'w') as f:
                    f.write(str(floor(fps)))
                break

            counter +=1

    def calc_fps(self, t_now: float, t0: float, n_frame: int) -> float:
        """
        Calculate FPS

        Parameters
        ----------

            - t_now (float): Current time
            - t0 (float): Initial time
            - n_frame (int): Number of frames

        Returns
        -------

            - float: FPS
        """
        div = (t_now - t0)

        if div > 0 :
            return n_frame / div
        else:
            return 0

    def print_features(self, frame: list, features_to_print: list) -> None:
        """
        Print features

        Parameters
        ----------

            - frame (list): Frame
            - features_to_print (list): Features to print

        Returns
        -------
            - None
        """

        for feature in features_to_print:
            cv2.putText(
                    frame,
                    "{}: {:.2f}".format(feature[0], feature[1]),
                    feature[2],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    feature[3],
                    2,
            )

    def run(self) -> None:
        """
        Run the real time system
        """

        # # Start video capture
        # video_frame = cv2.VideoCapture(self.camera)

        # # Set cv2 optimized to true, if it is not
        # if not cv2.useOptimized():
        #     try:
        #         cv2.setUseOptimized(True)
        #     except:
        #         print(
        #             "OpenCV optimization could not be set to True, the script may be slower than expected")

        frame = cv2.imread('./test_data/face.jpg')
        # Run the 3 subsystems in cascade
        counter = 0
        while True:
            # If there is not ret
            # ret, frame = video_frame.read()
            # if not ret:
            #     break

            # if the frame comes from webcam, flip it so it looks like a mirror.
            if self.camera == 0:
                frame = cv2.flip(frame, 2)
            
            # Run facuial features extraction model
            landmarks,frame_size = self.facial_features_extraction.run(frame)

            # Print landmarks
            if  self.print_landmarks:
                self.facial_features_extraction.show_eye_keypoints(
                    color_frame=frame, landmarks=landmarks, frame_size=frame_size)
            
            #Calculate FPS
            t_now = time.perf_counter()
            fps = self.calc_fps(t_now, self.t0, counter)

            fatigue_prediction = self.fatigue_detection_system.run(landmarks)
            avg_ear = self.fatigue_detection_system.get_avg_ear()
            avg_mar = self.fatigue_detection_system.get_avg_mar()
            perclos = self.fatigue_detection_system.get_perclos()
            pom = self.fatigue_detection_system.get_pom()
            poy = self.fatigue_detection_system.get_poy()

            features_to_print = [("EAR", avg_ear, (300, 30), (0, 255, 0)),("MAR", avg_mar, (300, 60), (0, 255, 0)),
                                ("PERCLOS", perclos, (300, 90), (0, 255, 0)),("POM", pom, (300,120), (0, 255, 0)),
                                ("POY", poy, (300,150), (0, 255, 0)),("Fatigue", fatigue_prediction, (10, 60), (0, 0, 255)),
                                ("FPS", fps, (300, 180), (0, 255, 0))]
            
            self.print_features(frame, features_to_print)
            if self.save_logs and avg_ear != 0 and avg_mar != 0:

                # Press "space" to save fatigue detected frame
                if cv2.waitKey(1) & 0xFF == ord(" "):
                    fatigue_prediction = 1
                else:
                    fatigue_prediction = 0
                
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                # COLS: timestamp;path_img;ear;mar
                path_img = f"./frames/{CATEGORY}/"

                # mkdir
                if not os.path.exists(path_img):
                    os.makedirs(path_img)

                full_path_img = "{}frame_{}.jpg".format(path_img, timestamp)

                # Save frame to disk
                cv2.imwrite(
                    full_path_img,
                    frame,
                )

                # Save ear and mar values for each frame, to make some alalysis
                with open(self.log_file.format(PARAMS_TO_LOG), "a+") as f:

                    row_to_write = ";".join(map(str,[timestamp, full_path_img, avg_ear, avg_mar,list(*landmarks), fatigue_prediction])) 
                    f.write(
                    row_to_write + "\n"
                    )

            # alert_result = self.alert_system.run(fatigue_prediction)
            # Save frame as a image
            cv2.imwrite('./data/face_metrics.jpg', frame)
            # cv2.imshow("Fatigue Detector", frame)

            counter +=1

            # Press "q" to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            break

        # Release video capture and close all windows when exiting
        cv2.destroyAllWindows()
        # video_frame.release()


if __name__ == "__main__":
    RealTime.run()