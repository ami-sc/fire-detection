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
    
                
#check this to understand how npwhere works

background = np.random.randint(0,255,[4,1,3]) #like a 4x1 image
img = np.random.randint(0,255,[4,1,3])
print(background)
print(img)

binary_mask = abs(background-img)>50
print(binary_mask)

new_background = np.where(binary_mask, img, background)
print(new_background)
"""

# Useful functions:
def outline():
    print("-------------------NEXT FRAME-------------------")

### PROGRAMM STARTS HERE ###

# constants:
initial_treshold = 50
a = 0.7 # small = very fast recovery ||| big = very slow recovery
c = 4 # small = doesn't detect small differrences ||| big = detects big differences


# initialization
video = cv2.VideoCapture(0) #put instead of 0 the path of the video
background = video.read()[1]
threshold = np.full(background.shape, initial_treshold)
red = np.full(background.shape, [0,0,255], dtype = np.uint8) #(1) this is needed only if you want to see where it changes

while True: 
    
    img = video.read()[1]
    cv2.imshow("camera", img)

    difference = abs(img-background)
    #a binary mask of the same shape of the img
    #the value[i,j] == true if abs(img-treshold) > 50, which means iff the image changes enough
    changes = difference > threshold 
    highlight = background = np.where(changes, red, img) #(1) red if true else value from camera
    cv2.imshow("changes", highlight) #(1)

    #compute the new background and tresholds as if we had to update all the pixels (is it possible to ptimize more? computing only if we had to compute it?)
    updated_background = background*a + (1-a)*img
    updated_treshold = threshold*a+(1-a)*(c*difference)

    #update the background and thresholds only if difference > threshold 
    background = np.where(changes, updated_background, background)
    threshold = np.where(changes, updated_treshold, threshold)
    #cv2.imshow("background", background)
    
    if  cv2.waitKey(1) & 0xFF == ord('q'):
        break