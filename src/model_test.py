import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define constants for eye aspect ratio and the threshold for detecting fatigue
EAR_THRESHOLD = 0.3
CONSECUTIVE_FRAMES_THRESHOLD = 48

# Initialize dlib's face detector (HOG-based) and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("path_to_shape_predictor_68_face_landmarks.dat")

# Initialize counters
frames_counter = 0
consecutive_frames = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        left_eye = shape[42:48]
        right_eye = shape[36:42]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < EAR_THRESHOLD:
            consecutive_frames += 1
            if consecutive_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                cv2.putText(frame, "Fatigue Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            consecutive_frames = 0
        
        cv2.putText(frame, "EAR: {:.2f}".format(avg_ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Fatigue Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()