## FPS Check

import cv2
import sys
import time


# Setup Opencv
cascPath = "class.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)



fps = 0
count = 0
start = time.time()


while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 0)
    width, height = frame.shape[:2]
    size = (width, height)
    
    if frame.all():
        #size reduce
        m = 2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cv2.resize(gray, (int(height / m), int(width / m)))
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x*m, y*m), (x*m+w*m, y*m+h*m), (0, 255, 0), 2)

    end = time.time()
    if (end - start > 2):
        fps = int(count / 2)
        start = time.time()
        count = 0
        print(fps)
    else:
        count += 1

    

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
 
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
