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
        self.show = None

    def define_rectangles(self):
        h_n = 20 #How many points in the heihgt
        w_n = 30 # How many points in the width
        h = self.shape[0]
        w = self.shape[1]
        width_points = np.linspace(0, w, w_n, dtype=int)
        height_points = np.linspace(0, h, h_n, dtype=int)
        self.rectangles = []
        for ii, i, in enumerate(width_points):
            if ii == 0: continue
            for ij, j in enumerate(height_points):
                if ij == 0: continue
               # top_left = (width_points[ii-1], height_points[ij-1])
              #  bottom_right = (i,j)
                self.rectangles.append((width_points[ii-1], height_points[ij-1], i,j)) #top left, bottom right
                



    def draw_rectangles(self):
        img = self.video.read()[1]
        for top_left, bottom_right in self.rectangles:
            cv2.rectangle(img, top_left, bottom_right, [255,0,0], 3)
            cv2.imshow("", img)
            if  cv2.waitKey(10) & 0xFF == ord('q'):
                break
        

    def rectangle_moving(self, moving_pixels):
        self.moving_rectangles = []
        for borders in self.rectangles:
            rectangle = moving_pixels[borders[0]:borders[2], borders[1]:borders[3]]
            if rectangle.shape[0] != 0 and rectangle.shape[1]!=0:
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
            img = self.video.read()[1]
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
            moving_pixels = np.any(changes_compared_to_background, axis = 2)
            print(changes_compared_to_background)
            cv2.imshow("changes", highlight) #(1)
            
            previous_image = img
            #cv2.imshow("background", background)

            self.rectangle_moving(moving_pixels)
        
            if  cv2.waitKey(10) & 0xFF == ord('q'):
                break



t = cv2.VideoCapture(0)
motion = MotionDetection(t, 80, 0.95, 4)
motion.fit()

