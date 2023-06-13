import cv2
from threading import Thread

class motiondetection():
    
    def __init__(self, video = False):
        # Initialize the video capture object
        if video != False:
            self.cap = cv2.VideoCapture(video)
        else:
            self.cap = cv2.VideoCapture(0)  # 0 represents the default camera

        # Read the first frame
        self.frame = self.cap.read()[1]
        self.prev_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame = cv2.GaussianBlur(self.prev_frame, (21, 21), 0)

    def fit(self, show = True):
        while True:
            # Read the current frame
            self.frame = self.cap.read()[1]
            
            # Convert the frame to grayscale and apply blur
            current_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            current_frame = cv2.GaussianBlur(current_frame, (21, 21), 0)
            
            # Calculate the absolute difference between the current and previous frame
            frame_diff = cv2.absdiff(self.prev_frame, current_frame)
            
            # Apply a threshold to create a binary image
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

            # Dilate the thresholded image to fill in the holes
            dilated = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours of the moving objects
            contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Iterate over the contours and draw rectangles around the moving objects
            for contour in contours:
                if cv2.contourArea(contour) < 500:  # Adjust the minimum contour area as needed
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Show the frames
            cv2.imshow("Motion Detection", self.frame)
            
            # Quit if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Update the previous frame
            self.prev_frame = current_frame

        # Release the video capture object and close the windows
        self.cap.release()
        cv2.destroyAllWindows()

t = motiondetection()

status = True
detect_alg = Thread(target = t.fit)
detect_alg.start()

c=0
while status!=False:
    c+=1
    print("Hello")
    if c==1000:
        status = False

status = False
detect_alg.join()
