import cv2
import datetime
import numpy as np
import time
import csv
import plyer

lowHSV = (45, 110, 0) #15 81 0
highHSV = (100, 255, 255) #(45, 255, 255)

mouth_cascade = cv2.CascadeClassifier('src\CascadeClassifiers\haar-cascade-files-master\haarcascade_mcs_mouth.xml')
glasses_cascade = cv2.CascadeClassifier('src\CascadeClassifiers\haar-cascade-files-master\haarcascade_eye_tree_eyeglasses.xml')

def intersecting(f,s):
    posWidth = min(f[2], s[2]) > max(f[0], s[0])
    posHeight = min(f[3], s[3]) > max(f[1], s[1])
    return posHeight and posWidth
        
last = datetime.datetime.now() - datetime.timedelta(seconds=5) # timeout starts 5 minutes ago so it starts recording when program starts

def recordDrink():
    global last
    if datetime.datetime.now() - last > datetime.timedelta(seconds=5):
        print("timeout end")
        with open("src/DrinkingTimes.csv", "a") as file:
            writer = csv.writer(file)
            now = datetime.datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print(dt_string)
            writer.writerow([dt_string])
            cv2.putText(frame,'Good Job!',(20,20),0,4,(50,255,0))
            last = now
            #file.close();
        
            

vid = cv2.VideoCapture(0)

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

    max_rect = []


    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            max_rect = [x,y,w,h]
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
            print("DRINKING DETECTED")
            recordDrink()
            #time.sleep(60000)
            

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