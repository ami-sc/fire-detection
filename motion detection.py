

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
            
                Threshold(x, n+1) --> aT(x,n) + (1-a)(c|P(x,n)- B(x,n)|)    // for stationary
                                  --> T(x,n)      // for moving
                                  
       where a is a positive real close to 1, and c is real number greater then 1. Higher c --> 
       higher thgreshold
       
    6. Estimated background is subtracted from current image to detect moving regions:
        
                |P(x,n) - B(x,n)| > T(x,n)
"""

def motion_detection(video, a=0.95, c=4, initial_treshold=50):

    # 1 initialization treshold image
    background = video.read()[1]
    threshold = np.full(background.shape, initial_treshold)
    red = np.full(background.shape, [0,0,255], dtype = np.uint8) #(1) this is needed only if you want to see where it changes

    # 2 take first frame as background image
    previous_image = video.read()[1]

    ######starting loop#######
    while True:     
        img = video.read()[1]
        #cv2.imshow("camera", img)

    # 3 find if pixel is moving
        difference_with_previous = abs(img-previous_image)
        changes = difference_with_previous > threshold  #binary mask

    # 4 update background
        updated_background = background*a + (1-a)*img
        background = np.where(changes, updated_background, background) #background if false, updated_background if true

    # 5 update threshold
        difference_with_background = abs(img-background)
        changes_compared_to_background = difference_with_background > threshold #binary mask
        updated_treshold = threshold*a+(1-a)*(c*difference_with_background)
        threshold = np.where(changes, updated_treshold, threshold)

    # 6 detecting moving regions
        highlight = np.where(changes_compared_to_background, red, img) #(1) red if true else value from camera
        cv2.imshow("changes", highlight) #(1)
        
        previous_image = img
        #cv2.imshow("background", background)
    
        if  cv2.waitKey(10) & 0xFF == ord('q'):
            break

"""# run the program with your camera
initial_treshold = 50
a = 0.95 # small = very fast recovery ||| big = very slow recovery
c = 4 # small = doesn't detect small differrences ||| big = detects big differences
video = cv2.VideoCapture(0) #put instead of 0 the path of the video
motion_detection(video)"""


"""
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