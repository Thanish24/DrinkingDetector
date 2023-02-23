import cv2
import time

vid = cv2.VideoCapture(0)

# hsv threshold
#HSV_MIN = cv2.Scalar(18, 40, 90)
#HSV_MAX = cv2.Scalar(27, 255, 255)

front_cascade = cv2.CascadeClassifier('src/CascadeClassifiers/cascade_front.xml')
upper_cascade = cv2.CascadeClassifier('src\CascadeClassifiers\haar-cascade-files-master\haarcascade_upperbody.xml')
mouth_cascade = cv2.CascadeClassifier('src\CascadeClassifiers\haar-cascade-files-master\haarcascade_mcs_mouth.xml')
glasses_cascade = cv2.CascadeClassifier('src\CascadeClassifiers\haar-cascade-files-master\haarcascade_eye_tree_eyeglasses.xml')



while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Apply grayscale filtering
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detection and face rectangle
    #faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    #faces_rect2 = haar_cascade2.detectMultiScale(gray_img, 1.1, 9)

    #front = front_cascade.detectMultiScale(gray_img, 1.3, 5)
    #upper = upper_cascade.detectMultiScale(gray_img, 1.5, 9)
    mouth = mouth_cascade.detectMultiScale(gray_img, 1.8, 1) # tested 1.3, 5 and 1.1,1 and 1.9, 1 and 1.5 and 2 and 1.7
    glasses = glasses_cascade.detectMultiScale(gray_img, 1.8, 2) # tested 1.3, 5 and 1.1,2 and 1.9,2 and 1.5 and 2 and 1.7

    largestA = 0
    largestA2 = 0
    largestA3 = 0

    #for (x, y, w, h) in front:
    #    area = w * h
    #    if area > largestA:
    #        largestA = area
    #        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)


    for (x, y, w, h) in glasses:
        area2 = w * h
        if area2 > largestA2:
            largestA2 = area2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    for (x, y, w, h) in mouth:
        area3 = w * h
        if area3 > largestA3:
            largestA3 = area3
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)


    #for (x, y, w, h) in back:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
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