import cv2
import time

vid = cv2.VideoCapture(1)

lowHSV = (15, 81, 0)
highHSV = (45, 255, 255) #(45, 255, 255)


mouth_cascade = cv2.CascadeClassifier('src\CascadeClassifiers\haar-cascade-files-master\haarcascade_mcs_mouth.xml')
glasses_cascade = cv2.CascadeClassifier('src\CascadeClassifiers\haar-cascade-files-master\haarcascade_eye_tree_eyeglasses.xml')



while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    threshold_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    filtered = cv2.inRange(threshold_img, lowHSV, highHSV)

    masked = cv2.bitwise_and(frame, frame, mask=filtered)


    # Apply grayscale filtering
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # run cascade detection
    mouth = mouth_cascade.detectMultiScale(gray_img, 1.9, 5) 

    glasses = glasses_cascade.detectMultiScale(gray_img, 1.8, 2) 


    for (x, y, w, h) in glasses:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    for (x, y, w, h) in mouth:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)


    # Display the resulting frame
    cv2.imshow('frame', masked)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.5)
    
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()