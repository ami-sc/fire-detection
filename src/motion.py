"""
Created on Thu Jun  8 14:14:08 2023

@author: mediolanum
"""

import numpy as np 
import cv2

class MotionDetection():
    def __init__(self, video, initial_treshold, a, c, grayscale = False):
        self.a = a
        self.c  = c
        self.shape = video.read(0)[1].shape
        self.video = video
        self.threshold = np.full(self.shape, initial_treshold)
        self.red = np.full(self.shape, [0,0,255], dtype = np.uint8) 

    def define_rectangles(self):
        h_n = 20  # number of rectangles in height
        w_n = 60  # number of rectangle in width
        h = self.shape[0]
        w = self.shape[1]

        width_points = np.linspace(0, w, w_n, dtype=int)
        height_points = np.linspace(0, h, h_n, dtype=int)

        width_start, height_start = np.meshgrid(width_points[:-1], height_points[:-1])
        width_end, height_end = np.meshgrid(width_points[1:], height_points[1:])

        self.rectangles = np.stack((width_start, height_start, width_end, height_end), axis=-1).reshape(-1, 4)





    def draw_rectangles(self):
        count = 0
        for u in self.rectangles:
            top_left, bottom_right = (u[0], u[1]),  (u[2], u[3])
            if self.moving_rectangles[count] == True: 
                
                cv2.rectangle(self.img, top_left, bottom_right, [255,0,0], 3)
            count += 1
        

    def rectangle_moving(self, moving_pixels):
        self.moving_rectangles = []
        for borders in self.rectangles:
                rectangle = moving_pixels[borders[0]:borders[2], borders[1]:borders[3]]
            #if rectangle.shape[0] != 0 and rectangle.shape[1]!=0:
                if rectangle.sum()>rectangle.size*0.2:
                    self.moving_rectangles.append(True)
                else:
                    self.moving_rectangles.append(False)
                  


    def fit(self):
        background = self.video.read()[1]
        previous_image = self.video.read()[1]
        threshold = self.threshold
        a = self.a
        c = self.c
        red = self.red

        self.define_rectangles()
        ######starting loop#######
        while True:     
            self.img = self.video.read()[1]
            #cv2.imshow("camera", img)

        # 3 find if pixel is moving
            difference_with_previous = abs(self.img-previous_image)
            changes = difference_with_previous > threshold  #binary mask

        # 4 update background
            updated_background = background*a + (1-a)*self.img
            background = np.where(changes, updated_background, background) #background if false, updated_background if true

        # 5 update threshold
            difference_with_background = abs(self.img-background)
            changes_compared_to_background = difference_with_background > threshold #binary mask
            updated_treshold = threshold*a+(1-a)*(c*difference_with_background)
            threshold = np.where(changes, updated_treshold, threshold)

        # 6 detecting moving regions
         #   highlight = np.where(changes_compared_to_background, red, img) #(1) red if true else value from camera
            moving_pixels = np.any(changes_compared_to_background, axis = 2)
      
           # cv2.imshow("changes", highlight) #(1)
            previous_image = self.img
            #cv2.imshow("background", background)
            
            self.rectangle_moving(moving_pixels)
           
            self.draw_rectangles()
            cv2.imshow("", self.img)
            if  cv2.waitKey(10) & 0xFF == ord('q'):
                break



t = cv2.VideoCapture(0)
motion = MotionDetection(t, 50, 0.95, 4)
motion.fit()


