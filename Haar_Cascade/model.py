import numpy as np
import cv2
f_cascade = cv2.CascadeClassifier("face.xml")
e_cascade = cv2.CascadeClassifier("eye.xml")
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if type(frame) is not str:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = f_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = e_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',frame)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break