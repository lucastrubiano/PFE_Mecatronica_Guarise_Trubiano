EAR_THRESHOLD = 0.30
MAR_THRESHOLD = 1.2
TIME_THRESHOLD = 0.5
PERCLOS_THRESHOLD = 0.7
POY_THRESHOLD = 3
YAWN_THRESHOLD = 0.7
FRAMES_PER_SECONDS = 34
CONSECUTIVE_FRAMES_THRESHOLD = 34
CONSECUTIVE_SEC_PERCLOS_THRESHOLD = 5
CONSECUTIVE_SEC_POM_THRESHOLD = 7
CONSECUTIVE_SEC_POY_THRESHOLD = 100
CONSECUTIVE_FRAMES_FPS = 200
EYES_LMS_NUMS = [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]
LEFT_EYE_LMS_NUMS = [33, 133, 160, 144, 158, 153]
RIGHT_EYE_LMS_NUMS = [362, 263, 385, 380, 387, 373]
LEFT_IRIS_NUM = 468
RIGHT_IRIS_NUM = 473
MOUTH_LMS_NUMS = [0, 267, 269, 270, 13, 14, 17, 402, 146, 405, 409, 415, 291, 37, 39, 40, 178, 308, 181, 310, 311, 312, 185, 314, 317, 318, 61, 191, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
OUTTER_LIP_LMS_NUMS = [0,37,39,40,185,61,146,91,181,84,17,314,405,321,375,91,409,270,269,267]
INNER_LIP_LMS_NUMS = [13,82,81,80,191,78,95,88,178,87,14,317,402,318,324,308,415,310,311,312]
CENTER_MOUTH_LMS_NUMS = [13,14]
BORDER_MOUTH_LMS_NUMS = [308,78]
FILE_FPS = ".FPS"
LANDMARKS_MODEL = "./models/shape_predictor_68_face_landmarks.dat"
DIR_LOGS = "logs"
MODEL_PATH = "./models/fatigue_detection_model.h5"
LOG_FILE = "./logs/{}.csv"

with open(FILE_FPS, 'r') as f:
    FPS = int(f.read())