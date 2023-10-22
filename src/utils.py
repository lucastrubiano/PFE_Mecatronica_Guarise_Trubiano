import cv2
import time
from config import CONSECUTIVE_FRAMES_FPS, FILE_FPS
from math import floor


def print_features(frame: list, features_to_print: list) -> None:
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

def set_fps(camera: int = 0, consecutive_frames_fps: float = CONSECUTIVE_FRAMES_FPS, file_fps: float = FILE_FPS) -> None:

    """
    Set FPS of the device

    Parameters
    ----------
        - camera (int, optional): Camera input. Defaults to 0.
        - consecutive_frames_fps (float, optional): Consecutive frames. Defaults to CONSECUTIVE_FRAMES_FPS.
        - file_fps (float, optional): File FPS. Defaults to FILE_FPS.

    Returns
    -------

        - None
    """

    video_frame = cv2.VideoCapture(camera)
    t0 = time.perf_counter()

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
        fps = calc_fps(t_now, t0, counter)

        if counter > consecutive_frames_fps:
            with open(file_fps, 'w') as f:
                f.write(str(floor(fps)))
            break

        counter +=1

@staticmethod
def calc_fps(t_now: float, t0: float, n_frame: int) -> float:
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