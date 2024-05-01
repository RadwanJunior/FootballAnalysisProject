import cv2

# Function that reads videos ---> just a stream of images
def read_video(video_path):
    # A videocapture object is used to get the video path and save it in the class
    cap = cv2.VideoCapture(video_path)
    # Array/list storing frames
    frames = []
    # Keeps looping while true
    while True:
        #Returns a flag whether the video has ended or not in the ret
        #Reads the next frame 
        ret, frame = cap.read()
        #If no next frame, video has ended and we break out of the loop
        if not ret:
            break
        #If true we append the frame to the list of frames, so we can get all the frames
        frames.append(frame)
    #Returns all the frames in a list
    return frames

