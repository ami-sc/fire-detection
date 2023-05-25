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

def find_if_pixel_moving(current_frame, previous_frame, Threshold):
    Status = []
    for i in range(len(current_frame)):
        a = []
        for j in range(len(current_frame[0])):          
            if abs(current_frame[i][j] - previous_frame[i][j]) > Threshold[i][j]:
                a.append(1)
            else: 
                a.append(0)
        Status.append(a)
    
    return Status



def Update_background(current_frame, Background, Status, Threshold):
    a = 0.95
    c = 1.3
    for i in range(len(current_frame)):
        for j in range(len(current_frame[0])):
            
            if Status[i][j] == 0:
                Threshold[i][j] = Threshold[i][j]*a + (1-a)*(c*abs(current_frame[i][j]-Background[i][j]))
            
            
    return Threshold
                
def Update_Threshold(current_frame, Background, Threshold, previous_frame):
    a = 0.95
    c = 2
    for i in range(len(current_frame)):
        for j in range(len(current_frame[0])):
            #if Status[i][j] == 0:
            if abs(current_frame[i][j] - previous_frame[i][j]) > Threshold[i][j]:
                Threshold[i][j] = Threshold[i][j]*a + (1-a)*(c*abs(current_frame[i][j]-Background[i][j]))
                Background[i][j] = Background[i][j]*a + (1-a)*current_frame[i][j]
            
    return (Threshold,Background)



def estimate_moving_regions(current_frame, Background, Threshold, image):

    
    for i in range(len(current_frame)):
        
        for j in range(len(current_frame[0])):
            a = current_frame[i][j] - Background[i][j]
            
            if abs(a) > Threshold[i][j]: 
             
                #current_frame[i][j] = False
                image[i][j] = (0,0,255)         
       
    return image #current_frame
            
    
def lol(image, moving_pixels): 
    
    for i in range(len(moving_pixels)):
        for j in range(len(moving_pixels[0])): 
            if moving_pixels[i][j] == False: 
                image[i][j] = (0,0,255)
    return image


#video = cv2.VideoCapture('C:/Users/mediolanum/Desktop/video/fire.mp4')
video = cv2.VideoCapture(0)
image = video.read()[1]
img,b,r = cv2.split(image)

Background = []
Threshold = np.full((img.shape[0], img.shape[1]), 50)

previous_frame = None

cv2.namedWindow("l", cv2.WINDOW_KEEPRATIO)
while True: 
    
    image = video.read()[1]
    img,b,r = cv2.split(image)

   
    img = np.asarray(img, dtype=np.int16)
   
    if len(Background) == 0: 
        Background = img
        previous_frame = img
        continue
    
    #Status = find_if_pixel_moving(img, previous_frame, Threshold)    
#    Background_updated = Update_background(img, Background, Status)
    Threshold_updated,Background_updated = Update_Threshold(img, Background, Threshold,previous_frame)
    moving_pixels = estimate_moving_regions(img, Background, Threshold, image)
    #oo = lol(image, moving_pixels)
    previous_frame = img
    Background = Background_updated
    Threshold = Threshold_updated
    
  
    
    cv2.imshow("l", moving_pixels)
    
    cv2.waitKey(100)
    
  
    print("-------------------NEXT FRAME-------------------")
    
cv2.destroyAllWindows()
    
    
        