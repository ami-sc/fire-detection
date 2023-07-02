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

    def define_rectangles(self, bigger_rectangles = 1):
        """This method defines the borders of the moving regions"""

        def find_overlapping_rectangles(bigger_rectangles, smaller_rectangles):
            """This is an helper function that is used only by this method
                it's aim is to find the index of the smaller rectangles that overlap a bigger rectangles
                In this way, each bigger rectangle is associated to certain amount of smaller rectangles"""
            overlapping_lists = []
            for big_idx, big_rect in enumerate(bigger_rectangles):
                overlapping_indices = []
                for small_idx, small_rect in enumerate(smaller_rectangles):
                    if (small_rect[0] < big_rect[0] + big_rect[2] and
                        small_rect[0] + small_rect[2] > big_rect[0] and
                        small_rect[1] < big_rect[1] + big_rect[3] and
                        small_rect[1] + small_rect[3] > big_rect[1]):
                        overlapping_indices.append(small_idx)
                overlapping_lists.append(overlapping_indices)
            return overlapping_lists
        
        h_n = 10  # number of rows of smaller rectangles
        w_n = 10  # number of columns of smaller rectangles
        h = self.shape[0] # the shape of the image
        w = self.shape[1]

        width_points = np.linspace(0, w, w_n, dtype=int) #take w_n points from 0 to w
        height_points = np.linspace(0, h, h_n, dtype=int)#take h_n points from 0 to h
        width_start, height_start = np.meshgrid(width_points[:-1], height_points[:-1]) #generate all the top left angles of each rectangle
        width_end, height_end = np.meshgrid(width_points[1:], height_points[1:]) #generate all bottom right angles of each rectangle
        #self.rectnagles = [[X_top, Y_top, X_bottom, Y_bottom], [...], ...]
        self.rectangles = np.stack((width_start, height_start, width_end, height_end), axis=-1).reshape(-1, 4) 
        
        if bigger_rectangles!=0: # it could also be done recursively, having always bigger rectangles. degree of rectangles = "degree of smaller rectanlge +1"
            # self.rectangles_m = [[[X1_top, Y1_top, X1_bottom, Y1_bottom], [...], ...],[[X2_top, Y2_top, X2_bottom, Y2_bottom], [...], ...]]]
            self.rectangles_m = [] # 1 is for rectangles of degree 1, 2 is for rectangels of degree 2
            # self.small_rectangles_index_m = [[[1,5,...], [...], ...], [[4,0,...], [...], ...], [...], ...]
            # the ith list represents the rectangles of degree i. the elements of the ith list represent for each bigger rectangle the indices of the 
            # smaller rectangles overlapping the bigger rectangle
            self.small_rectangles_index_m = []

            for i in range(1,bigger_rectangles+1):
                #same procedure
                # if i = 1 the rectangles will  be double the size of smallest rectangles
                # if i = 2 the rectangles will  be 4x the size of smallest rectangles
                # ...
                width_points = np.linspace(0, w, w_n//(2*i), dtype=int)
                height_points = np.linspace(0, h, h_n//(2*i), dtype=int)
                width_start, height_start = np.meshgrid(width_points[:-1], height_points[:-1])
                width_end, height_end = np.meshgrid(width_points[1:], height_points[1:])
                bigger_rectangles = np.stack((width_start, height_start, width_end, height_end), axis=-1).reshape(-1, 4)
                # end of generation of regtangles of ith degree
                self.rectangles_m.append(bigger_rectangles)
                self.small_rectangles_index_m.append(find_overlapping_rectangles(bigger_rectangles,self.rectangles))
                
    def get_model_input(self, save = True):
        self.model_input = []
        for u, moving in enumerate(self.moving_rectangles):
            if moving:
                rectangle = self.rectangles[u]
                self.model_input.append(self.img[rectangle[0]:rectangle[2]][rectangle[1]:rectangle[3]][:])
        if save:
            for i,image in enumerate(self.model_input):
                cv2.imwrite(f"../data/model_input/{self.i}{i}input.jpg", image)
        return self.model_input

    def draw_rectangles(self):
        #draw bigger rectangles
        for degree, moving_regctangles_degree in enumerate(self.moving_rectangles_index_m):
            for rectangle_index in moving_regctangles_degree:
                u = self.rectangles_m[degree][rectangle_index]
                top_left, bottom_right = (u[0], u[1]),  (u[2], u[3])
                cv2.rectangle(self.img, top_left, bottom_right, [255,255,0], 3)# draw the borders of the rectangle
        #draw smaller rectangles
        for i in self.moving_rectangles_index:
            u = self.rectangles[i]
            top_left, bottom_right = (u[0], u[1]),  (u[2], u[3])
            cv2.rectangle(self.img, top_left, bottom_right, [255,0,0], 3)# draw the borders of the rectangle



    def rectangle_moving(self, moving_pixels):
        self.moving_rectangles_index = [] # list with the index of all smaller moving rectangles
        for i, borders in enumerate(self.rectangles):# for every rectangle(coordinates)
                rectangle = moving_pixels[borders[1]:borders[3], borders[0]:borders[2]] # take the pixels in the region
                if rectangle.sum()>rectangle.size*0.2: # if more that 20% of the pixels are moving
                    self.moving_rectangles_index.append(i) # then append the index of the rectangle to the list of moving rectangles index

        self.moving_rectangles_index = np.array(self.moving_rectangles_index)

        self.moving_rectangles_index_m = [] #list with the index of all bigger moving rectangles
        for i, small_rectangles_index in enumerate(self.small_rectangles_index_m): # for every degree of bigger rectangle
            contained = set() #contains the index of the rectangles of (i-1)th degree contained by the rectangles of (i)th degree
            self.moving_rectangles_index_m.append([]) #create a empty list of the rectangles of ith degree
            for j, smaller_rectangles_in_i in enumerate(small_rectangles_index): # for every rectangle of ith degree
                common = np.intersect1d(np.array(smaller_rectangles_in_i), self.moving_rectangles_index) # take the moving rectangles of (i-1)th degree contained in the ith degree rectangle
                if len(common)/len(smaller_rectangles_in_i)>0.25: # if more than 25% of rectangles of (i-1)th degree contained are moving
                    self.moving_rectangles_index_m[i].append(j) # then the ith degree rectangle is moving
                    contained.update(common) # put the rectangles of (i-1)th degree in the set of moving and contained rectangles of (i-1)th degree
        self.moving_rectangles_index = np.setdiff1d(self.moving_rectangles_index, list(contained))#the rectangles of (i-1)th that are moving and are not contained
        
    def fetch_model_input(self):
        """transforms the moving regions into images for the model"""
        images = []
        for degree, moving_regctangles_degree in enumerate(self.moving_rectangles_index_m):
            for rectangle_index in moving_regctangles_degree:
                u = self.rectangles_m[degree][rectangle_index]
                top_left, bottom_right = (u[0], u[1]),  (u[2], u[3])
                image = self.img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                images.append(image)

        for i in self.moving_rectangles_index:
            u = self.rectangles[i]
            top_left, bottom_right = (u[0], u[1]),  (u[2], u[3])
            image = self.img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            images.append(image)
        i = 0
        for image in images:
            cv2.imwrite(f"../data/model_input/g{i}.jpg", image)
            i+=1
        return images
    
            


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
        self.i=-1
        while self.run:    
            self.i+=1 
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

            self.fetch_model_input()
            cv2.imshow("", self.img)
            if  cv2.waitKey(10) & 0xFF == ord('q'):
                break

t = MotionDetection()
t.fit()



