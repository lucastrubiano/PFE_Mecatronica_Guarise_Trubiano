from __future__ import annotations

import time
from math import floor

import cv2
import numpy as np

from config import CONSECUTIVE_FRAMES_FPS
from config import FILE_FPS


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
            f'{feature[0]}: {feature[1]}' if type(
                feature[1],
            ) == str else f'{feature[0]}: {feature[1]:.2f}',
            feature[2],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            feature[3],
            2,
        )


def set_fps(camera: int = 0, consecutive_frames_fps: float = CONSECUTIVE_FRAMES_FPS, file_fps: str = FILE_FPS) -> None:
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
                'OpenCV optimization could not be set to True, the script may be slower than expected',
            )

    counter = 0
    # Run the 3 subsystems in cascade
    while True:

        # If there is not ret
        ret, _ = video_frame.read()
        if not ret:
            break

        # Calculate FPS
        t_now = time.perf_counter()
        fps = calc_fps(t_now, t0, counter)

        if counter > consecutive_frames_fps:
            with open(file_fps, 'w') as f:
                f.write(str(floor(fps)))
            break

        counter += 1


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

    if div > 0:
        return n_frame / div
    else:
        return 0


def isRotationMatrix(R, precision=1e-4):
    """
    Checks if a matrix is a rotation matrix
    :param R: np.array matrix of 3 by 3
    :param precision: float
        precision to respect to accept a zero value in identity matrix check (default is 1e-4)
    :return: True or False
        Return True if a matrix is a rotation matrix, False if not
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < precision


def rotationMatrixToEulerAngles(R, precision=1e-4):
    '''
    Computes the Tait–Bryan Euler (XYZ) angles from a Rotation Matrix.
    Also checks if there is a gymbal lock and eventually use an alternative formula
    :param R: np.array
        3 x 3 Rotation matrix
    :param precision: float
        precision to respect to accept a zero value in identity matrix check (default is 1e-4)
    :return: (yaw, pitch, roll) tuple of float numbers
        Euler angles in radians in the order of YAW, PITCH, ROLL
    '''

    # Calculates Tait–Bryan Euler angles from a Rotation Matrix
    assert (isRotationMatrix(R, precision))  # check if it's a Rmat

    # assert that sqrt(R11^2 + R21^2) != 0
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < precision

    if not singular:  # if not in a singularity, use the standard formula
        x = np.arctan2(R[2, 1], R[2, 2])  # atan2(R31, R33) -> YAW, angle PSI

        # atan2(-R31, sqrt(R11^2 + R21^2)) -> PITCH, angle delta
        y = np.arctan2(-R[2, 0], sy)

        z = np.arctan2(R[1, 0], R[0, 0])  # atan2(R21,R11) -> ROLL, angle phi

    else:  # if in gymbal lock, use different formula for yaw, pitch roll
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])  # returns YAW, PITCH, ROLL angles in radians


def draw_pose_info(frame, img_point, point_proj, roll=None, pitch=None, yaw=None):
    """
    Draw 3d orthogonal axis given a frame, a point in the frame, the projection point array.
    Also prints the information about the roll, pitch and yaw if passed

    :param frame: opencv image/frame
    :param img_point: tuple
        x,y position in the image/frame for the 3d axis for the projection
    :param point_proj: np.array
        Projected point along 3 axis obtained from the cv2.projectPoints function
    :param roll: float, optional
    :param pitch: float, optional
    :param yaw: float, optional
    :return: frame: opencv image/frame
        Frame with 3d axis drawn and, optionally, the roll,pitch and yaw values drawn
    """
    frame = cv2.line(
        frame, img_point, tuple(
            point_proj[0].ravel().astype(int),
        ), (255, 0, 0), 3,
    )
    frame = cv2.line(
        frame, img_point, tuple(
            point_proj[1].ravel().astype(int),
        ), (0, 255, 0), 3,
    )
    frame = cv2.line(
        frame, img_point, tuple(
            point_proj[2].ravel().astype(int),
        ), (0, 0, 255), 3,
    )
    if roll is not None and pitch is not None and yaw is not None:
        cv2.putText(
            frame, 'Roll:' + str(round(roll, 0)), (500, 50),
            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, 'Pitch:' + str(round(pitch, 0)), (500, 70),
            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, 'Yaw:' + str(round(yaw, 0)), (500, 90),
            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return frame
