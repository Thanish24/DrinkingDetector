import cv2
import time
import numpy as np

vid = cv2.VideoCapture(0)

lowHSV = (45, 110, 0) #15 81 0
highHSV = (100, 255, 255) #(45, 255, 255)


mouth_cascade = cv2.CascadeClassifier('src\CascadeClassifiers\haar-cascade-files-master\haarcascade_mcs_mouth.xml')
glasses_cascade = cv2.CascadeClassifier('src\CascadeClassifiers\haar-cascade-files-master\haarcascade_eye_tree_eyeglasses.xml')

def intersecting(f,s):
    posWidth = min(f[2], s[2]) > max(f[0], s[0])
    posHeight = min(f[3], s[3]) > max(f[1], s[1])
    return posHeight and posWidth



while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    # detect bottle
    threshold_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    threshold_img = cv2.blur(threshold_img, (20,20))

    #threshold_img = cv2.erode(threshold_img, (50,50)) 


    filtered = cv2.inRange(threshold_img, lowHSV, highHSV)

    #filtered = cv2.bitwise_not(filtered, filtered)

    masked = cv2.bitwise_and(frame, frame, mask=filtered)

    contours, _ = cv2.findContours(filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0;
    max_rect = []

    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 60 or rect[3] < 60: continue
        if rect[2] * rect[3] > max_area:
            max_rect = rect
        x,y,w,h = max_rect
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame,'bottle Detected',(x+w+10,y+h),0,0.5,(255,0,0))


    # detect eyes and mouth

    # Apply grayscale filtering
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run cascade detection
    mouth = mouth_cascade.detectMultiScale(gray_img, 1.9, 5) 

    #glasses = glasses_cascade.detectMultiScale(gray_img, 1.8, 2) 

    mouth_area = []

    for (x, y, w, h) in mouth:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        mouth_area = [x, y, h*4, w*2]
    

    if mouth_area != [] and max_rect != []:
        cv2.rectangle(frame, (mouth_area[0] - int(0.5*mouth_area[2]), mouth_area[1]- int(0.5*mouth_area[3])), (mouth_area[0]+ mouth_area[2], mouth_area[1] + mouth_area[3]), (0,255,0), thickness=2)

        bottle_coords = [max_rect[0], max_rect[1], max_rect[0] + max_rect[2], max_rect[1] + max_rect[3]]
        mouth_area_coords = [mouth_area[0], mouth_area[1], mouth_area[0] + mouth_area[2], mouth_area[1] + mouth_area[3]]
        
        if intersecting(bottle_coords, mouth_area_coords):
            print("DRINKING")
            


    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #time.sleep(0.5)
    
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()