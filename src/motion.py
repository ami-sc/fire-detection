import numpy as np 
import cv2

class MotionDetection():

    def __init__(self, video_path=None, initial_treshold=50, a=0.95, c=4, grayscale = False):
        if video_path == None:
            self.video = cv2.VideoCapture(0) #use camera by default
        else:
            self.video = cv2.VideoCapture(video_path) #if a path is given, take the video from that path

        self.run = False

        #parameters for the algorithm
        self.a = a
        self.c  = c
        self.shape = self.video.read(0)[1].shape
        self.threshold = np.full(self.shape, initial_treshold)
        self.red = np.full(self.shape, [0,0,255], dtype = np.uint8) 


    def define_rectangles(self):
        #this function defines the border of the rectangles
        h_n = 20  # number of rows of rectangles
        w_n = 50  # number of columns of rectangles
        h = self.shape[0]
        w = self.shape[1]

        width_points = np.linspace(0, w, w_n, dtype=int)
        height_points = np.linspace(0, h, h_n, dtype=int)

        width_start, height_start = np.meshgrid(width_points[:-1], height_points[:-1])
        width_end, height_end = np.meshgrid(width_points[1:], height_points[1:])

        self.rectangles = np.stack((width_start, height_start, width_end, height_end), axis=-1).reshape(-1, 4)

    def draw_rectangles(self):
        count = 0 #used to identify the rectangle
        for u in self.rectangles: # for every rectangle
            top_left, bottom_right = (u[0], u[1]),  (u[2], u[3])
            if self.moving_rectangles[count] == True: 
                # if that region is moving
                cv2.rectangle(self.img, top_left, bottom_right, [255,0,0], 3)# draw the borders of the rectangle
            count += 1
        
    def rectangle_moving(self, moving_pixels):
        self.moving_rectangles = [] # [True, False, ...] if index i==True then region i is moving
        for borders in self.rectangles:# for every rectangle
                # take the pixels in the region and see if they are moving
                rectangle = moving_pixels[borders[1]:borders[3], borders[0]:borders[2]]
            #if rectangle.shape[0] != 0 and rectangle.shape[1]!=0:
                if rectangle.sum()>rectangle.size*0.2: # if more that 20% of the pixels are moving
                    self.moving_rectangles.append(True) # then we say that the region is moving
                else:
                    self.moving_rectangles.append(False)

    def stop(self):
        self.run = False

    def fit(self):
        background = self.video.read()[1]
        previous_image = self.video.read()[1]
        threshold = self.threshold
        a = self.a
        c = self.c
        red = self.red

        self.define_rectangles()
        self.run=True
        ######starting loop#######
        while self.run:     
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

t = MotionDetection()
t.fit()



