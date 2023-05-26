# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:29:44 2023

@author: mediolanum
"""

import cv2 
import numpy as np


"""
# ALGORITHMO: 
    
    1. Initialize Threshold image (pretermined value)
    2. Take first frame as a Background image, continue to the next frame
    3. Find if pixel is moving or not using the formula: P(x,n) - P(x,n-1) > Threshold(x,n)
       where x is position of pixel, n is current frame and n-1 is previous frame 
    4. Update Background: 
            
                Background(x,n+1) --> aB(x,n) + (1-a)P(x,n)    //for stationary pixel
                                  --> B(x,n)     // for moving pixel
                                  
    5. Update Threshold: 
            
                Threshold(x, n+1) --> aT(x,n) + (1-a)(c| P(x,n)- B(x,n)|)    // for stationary
                                  --> T(x,n)      // for moving
                                  
       where a is a positive real close to 1, and c is real number greater then 1. Higher c --> 
       higher thgreshold
       
    6. Estimated background is subtracted from current image to detect moving regions:
        
                |P(x,n) - B(x,n)| > T(x,n)
    
"""




                
def movement_detection_and_updates(current_frame, Background, Threshold, previous_frame, image):
    a = 0.95
    c = 2
    for i in range(len(current_frame)):  
        for j in range(len(current_frame[0])):
          
                
            # this is moving region estimation (on image it draws red pixels if movement detected)
           
            if abs( current_frame[i][j] - Background[i][j]) > Threshold[i][j]: 
                #current_frame[i][j] = False
                image[i][j] = (0,0,255)  
                
            # updates Threshold and Backgroun 
            current_frame_a = current_frame[i][j] 
            Threshold_a = Threshold[i][j]
            Background_a = Background[i][j]
         
            if abs(current_frame_a-previous_frame[i][j] ) > Threshold_a:
                Threshold_a = Threshold_a*a+(1-a)*(c*abs(current_frame[i][j] - Background[i][j]))
                Background_a = Background_a*a + (1-a)*current_frame_a
       
            
    return (Threshold,Background, image)



### PROGRAMM STARTS HERE ###


# reading video and splitting image into chanels
video = cv2.VideoCapture('C:/Users/mediolanum/Desktop/video/fire.mp4')
image = video.read()[1]
img,b,r = cv2.split(image)


# Initializing Threshold, Background and previous_frame
Threshold = np.full((img.shape[0], img.shape[1]), 50)
Background = img
previous_frame = img



cv2.namedWindow("l", cv2.WINDOW_KEEPRATIO)
while True: 
    
    image = video.read()[1]
    img,b,r = cv2.split(image)
    
    # converting array to np.int16 as later substracting and adding gives overflow error, 
    # Im trying to find the other way 
    img = np.asarray(img, dtype=np.int16)

   
    
        
    # Calling one function that will firstly compute if pixels are different from 
    # previous frame, then it will update threshold and background
    Threshold_updated,Background_updated, moving_pixels = movement_detection_and_updates(img, Background, Threshold,previous_frame, image)
    
    
    # updating 
    previous_frame = img
    Background = Background_updated
    Threshold = Threshold_updated
    
  
    
    cv2.imshow("l", moving_pixels)
    
    cv2.waitKey(10)
    
  
    print("-------------------NEXT FRAME-------------------")
    
cv2.destroyAllWindows()
    
        