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

# Function that takes outputted video frames as a list, as well as a path to save the video and saves the video
def save_video(output_video_frames, output_video_path):
    #We define an output format as XVID
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #Define a videowriter that takes in the video path(string), output video type, number of frames per second, output video frame width and height
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    #Loop over each frame and write each frame to the video writer
    for frame in output_video_frames:
        out.write(frame)
    #Once done release the videowriter
    out.release()
