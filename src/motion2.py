import cv2

fitting = False

def stop():
    #stops fitting function if it's running
    print("stop function")
    global fitting

    if fitting == True:
        print("fitting = False")
        fitting = False


def fit(video = False, toshow = True):
    global fitting
    print("fitting = True")
    fitting = True
    print(fitting)
    print(video)

    # By default it will use the camera
    print("initializing cap")
    if video != False:# if a video to use is given, it will use it
        cap = cv2.VideoCapture(video)
    else:
        print("video = False")
        cap = cv2.VideoCapture(0)

    # initializing first variables
    print("reading first frame")
    prev_frame = cap.read()[1] #first frame
    print(prev_frame.shape)
    print("manipulating first frame")
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0) #converting to grayscale and blurring

    while fitting: #until stop() is called run this
        # Read the current frame
        print("reading i frame")
        frame = cap.read()[1]
        
        # Convert the frame to grayscale and apply blur
        print("manipulating i frame")
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_frame = cv2.GaussianBlur(current_frame, (21, 21), 0)
        
        # Calculate the absolute difference between the current and previous frame
        frame_diff = cv2.absdiff(prev_frame, current_frame)
        
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
            if toshow:
                #if the running algorithm has to be showm, draw the rectangles representing moving areas
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if toshow:
            # Show the running algorithm
            cv2.imshow("Motion Detection", frame)
        
        # Update the previous frame
        prev_frame = current_frame
        
        # Quit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop()
            break
        print("end of iteration")
    print("destroying all windows and releasing cap")
    cv2.destroyAllWindows() # destroys cv2 window
    cap.release()
        

    
