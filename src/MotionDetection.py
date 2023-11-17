"""
**Motion Detection**

Detect moving regions in a given video.
"""


import cv2
import numpy as np
import xgboost as xgb

from WaveletFeatureExtraction import getWaveletFeatures


class MotionDetection:

    def __init__(self, video_path=None, a=0.95, c=2, initial_threshold=40):

        if video_path is None:
            # Use camera by default.
            self.video = cv2.VideoCapture(0)
        else:
            # If a path is given, take the video from that path.
            self.video = cv2.VideoCapture(video_path)

        # Load the model.
        self.model = xgb.Booster()
        self.model.load_model("./models/XGBoost_Tuned_Model.json")

        self.run = False

        self.contained = []
        self.img = None
        self.copy_img = 1
        self.rectangles = []
        self.rectangles_m = []
        self.small_rectangles_index_m = []
        self.moving_rectangles_index = []

        # List with the index of all bigger moving rectangles.
        self.moving_rectangles_index_m = []

        # Parameters for the motion detection algorithm.
        self.a = a
        self.c = c

        # Retrieve the shape of the input image.
        self.shape = self.video.read(0)[1].shape
        # Threshold matrix of equivalent size.
        self.threshold = np.full(self.shape, initial_threshold)
        # Red-colored matrix of equivalent size.
        self.red = np.full(self.shape, [0, 0, 255], dtype=np.uint8)

    def define_rectangles(self, bigger_rectangles=1):
        """
        This method defines the borders of the moving regions.
        """

        def find_overlapping_rectangles(bg_rectangles, smaller_rectangles):
            """
            This is a helper function used only by this method. Its aim is to
            find the index of the smaller rectangles that overlap bigger
            rectangles. In this way, each bigger rectangle is associated to
            a certain amount of smaller rectangles
            """

            overlapping_lists = []
            for big_idx, big_rect in enumerate(bg_rectangles):
                overlapping_indices = []
                for small_idx, small_rect in enumerate(smaller_rectangles):
                    if (small_rect[0] < big_rect[0] + big_rect[2] and
                            small_rect[0] + small_rect[2] > big_rect[0] and
                            small_rect[1] < big_rect[1] + big_rect[3] and
                            small_rect[1] + small_rect[3] > big_rect[1]):
                        overlapping_indices.append(small_idx)
                overlapping_lists.append(overlapping_indices)
            return overlapping_lists

        # Number of rows of smaller rectangles.
        h_n = 15
        # Number of columns of smaller rectangles.
        w_n = 15

        # Shape of the image.
        h = self.shape[0]
        w = self.shape[1]

        # Take w_n points from 0 to w.
        width_points = np.linspace(0, w, w_n, dtype=int)
        # Take h_n points from 0 to h.
        height_points = np.linspace(0, h, h_n, dtype=int)
        # Generate all the top left angles of each rectangle.
        width_start, height_start = np.meshgrid(
            width_points[:-1], height_points[:-1])
        # Generate all bottom right angles of each rectangle.
        width_end, height_end = np.meshgrid(
            width_points[1:], height_points[1:])

        self.rectangles = np.stack(
            (width_start, height_start, width_end, height_end),
            axis=-1).reshape(-1, 4)

        # Could also be done recursively, having always bigger rectangles.
        # degree of rectangles = "degree of smaller rectangle +1"
        if bigger_rectangles != 0:
            # self.rectangles_m = [[[X1_top, Y1_top, X1_bottom, Y1_bottom],
            #                [...], ...],
            #                [[X2_top, Y2_top, X2_bottom, Y2_bottom],
            #                [...], ...]]]

            # 1 is for rectangles of degree 1, 2 is for rectangles of degree 2.

            # self.small_rectangles_index_m = [[[1,5,...], [...], ...],
            #                            [[4,0,...], [...], ...], [...], ...]

            # The ith list represents the rectangles of degree i.
            # The elements of the ith list represent, for each bigger rectangle,
            # the indices of the smaller rectangles overlapping the bigger
            # rectangle.

            for i in range(1, bigger_rectangles + 1):
                # Same procedure...
                # If i = 1 the rectangles will be double the size of the
                # smallest rectangles.
                # If i = 2 the rectangles will be 4x the size of smallest
                # rectangles.

                width_points = np.linspace(0, w, w_n // (2 * i), dtype=int)
                height_points = np.linspace(0, h, h_n // (2 * i), dtype=int)
                width_start, height_start = np.meshgrid(
                    width_points[:-1], height_points[:-1])
                width_end, height_end = np.meshgrid(
                    width_points[1:], height_points[1:])
                bigger_rectangles = np.stack(
                    (width_start, height_start, width_end, height_end),
                    axis=-1).reshape(-1, 4)

                # End of generation of rectangles of ith degree.
                self.rectangles_m.append(bigger_rectangles)
                self.small_rectangles_index_m.append(
                    find_overlapping_rectangles(bigger_rectangles,
                                                self.rectangles))

    def draw_rectangles(self):

        # Draw bigger rectangles.
        self.copy_img = self.img.copy()
        for degree, moving_rectangles_degree in enumerate(
                self.moving_rectangles_index_m):

            for rectangle_index in moving_rectangles_degree:

                u = self.rectangles_m[degree][rectangle_index]
                top_left, bottom_right = (u[0], u[1]), (u[2], u[3])
                image = self.img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                # Get features for rectangle.
                fts = np.array([getWaveletFeatures(image)])
                # Predict using features.
                a = self.model.predict(xgb.DMatrix(fts))

                # Draw the borders of the rectangle.
                cv2.rectangle(self.copy_img, top_left, bottom_right, [
                    255, 255, 0], 3)

                # Show prediction score.
                cv2.putText(
                    self.copy_img, f"{a[0]:.2f}", top_left,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

        # Draw smaller rectangles.
        for i in self.moving_rectangles_index:

            u = self.rectangles[i]
            top_left, bottom_right = (u[0], u[1]), (u[2], u[3])
            image = self.img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Get features for rectangle.
            fts = np.array([getWaveletFeatures(image)])
            # Predict using features.
            a = self.model.predict(xgb.DMatrix(fts))

            # Draw the borders of the rectangle.
            cv2.rectangle(self.copy_img, top_left, bottom_right, [
                255, 0, 0], 3)

            # Show prediction score.
            cv2.putText(self.copy_img, f"{a[0]:.2f}", top_left,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

    def rectangle_moving(self, moving_pixels):

        # List with the index of all smaller moving rectangles.
        self.moving_rectangles_index = []

        for i, borders in enumerate(self.rectangles):
            # Take the pixels in the region.
            rectangle = moving_pixels[borders[1]:borders[3], borders[0]:borders[2]]

            # If more that 20% of the pixels are moving, then append the index
            # of the rectangle to the list of moving rectangles index.
            if rectangle.sum() > rectangle.size * 0.2:
                self.moving_rectangles_index.append(i)

        self.moving_rectangles_index = np.array(self.moving_rectangles_index)

        for i, small_rectangles_index in enumerate(
                self.small_rectangles_index_m):

            # Contains the index of the rectangles of (i-1)th degree contained
            # by the rectangles of (i)th degree.
            self.contained = set()

            # Create an empty list of the rectangles of ith degree.
            self.moving_rectangles_index_m.append([])

            # For every rectangle of ith degree.
            for j, smaller_rectangles_in_i in enumerate(small_rectangles_index):
                # Take the moving rectangles of (i-1)th degree contained in the
                # ith degree rectangle.
                common = np.intersect1d(
                    np.array(smaller_rectangles_in_i),
                    self.moving_rectangles_index)

                # If more than 25% of rectangles of (i-1)th degree contained
                # are moving...
                if len(common) / len(smaller_rectangles_in_i) > 0.2:
                    # ...then the ith degree rectangle is moving.
                    self.moving_rectangles_index_m[i].append(j)

                    # Put the rectangles of (i-1)th degree in the set of moving
                    # and contained rectangles of (i-1)th degree.
                    self.contained.update(common)

        # The rectangles of (i-1)th that are moving and are not contained.
        self.moving_rectangles_index = np.setdiff1d(
            self.moving_rectangles_index, list(self.contained))

    def stop(self):
        self.run = False

    def fit(self):

        cv2.namedWindow("Video", cv2.WINDOW_KEEPRATIO)

        background = self.video.read()[1]
        previous_image = self.video.read()[1]
        threshold = self.threshold
        a = self.a
        c = self.c

        self.define_rectangles()
        self.run = True

        # Start loop.
        i = -1
        while self.run:
            i += 1

            self.img = self.video.read()[1]

            # Find if pixel is moving.
            difference_with_previous = abs(self.img - previous_image)
            changes = difference_with_previous > threshold  # binary mask

            # Update background.
            updated_background = background * a + (1 - a) * self.img
            # background if false, updated_background if true
            background = np.where(changes, updated_background, background)

            # Update threshold.
            difference_with_background = abs(self.img - background)
            changes_compared_to_background = difference_with_background > threshold  # binary mask
            updated_threshold = threshold * a + (1 - a) * (
                    c * difference_with_background)
            threshold = np.where(changes, updated_threshold, threshold)

            # Detecting moving regions.
            moving_pixels = np.any(changes_compared_to_background, axis=2)

            previous_image = self.img

            self.rectangle_moving(moving_pixels)

            self.draw_rectangles()
            cv2.imshow("Video", self.copy_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
