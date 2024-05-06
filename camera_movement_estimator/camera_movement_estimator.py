import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        #If it moves at least 5, it is statistically significant for us to consider
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15,15),
            #Can downscale the image up to twice
            maxLevel = 2,
            #Stopping criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        #Put features in init to meake things cleaner
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        #Top banner
        mask_features[:,0:20] = 1
        #Bottom banner
        mask_features[:,900:1050] = 1

        self.features = dict(
            #Max amount of corners we can use for goodfeatures
            maxCorners = 100,
            #The higher the level the better the features but the lesser the amount of features you can get
            qualityLevel = 0.3,
            #Min distance between the features
            minDistance = 3,
            #Block size of the features
            blockSize = 7,
            #Where to extract the features from (from mask)
            mask = mask_features,


        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        #Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        #Read the stub when we have it
        #This will be the camera movement for each frame being stored in this array (Movement of x and y multiplied by the lenght of frames we have)
        camera_movement = [[0,0]]*len(frames)

        #convert the image into a grey image
        #Call all previous frames as old
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # 
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        #Will test and calculate the difference in movement between the 2 frame's features and decide whether there is distance between the camera movement or not
        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)
            #We need the max distance, because each frame will have multiple features and we want the max distance any two features
            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            #To enumerate over any two list, we have to zip them first
            for i, (new,old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            #Overide old gray with the frame gray
            old_gray = frame_gray.copy()

            #utilize stubs for pickle
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(camera_movement,f)
        
        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            #To not contaminate frames inputted to funciton, make a copy
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), cv2.FILLED)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement: .2f}" , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement: .2f}" , (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

            output_frames.append(frame)
        
        return output_frames



