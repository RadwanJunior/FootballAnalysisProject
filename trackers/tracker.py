#Tracker to trace bounding box locations
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    #Gets called when we initialize the class
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
    # Ball interpolation function
    def interpolate_ball_positions(self, ball_positions):
        #Convert ball positions into a panda data frame, but must be convereted to a list first

        ball_positions = [x.get(1,{}).get("bbox", []) for x in ball_positions]
        #Convert to panda data frame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        #Edge case if the missing detection is the first one, we can do this by replicating the nearest detection that we can find
        #Basically backfill it
        df_ball_positions = df_ball_positions.bfill()

        #Return ball postitions back to original format, list of lists
        # loops over all the things in df_ball positions then it puts it into a dictionary with 1 as track_id and the value is a dictionary of bounding boxes and the value for that will be x
        ball_positions = [{1: {"bbox": x}}for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames):
        #Add a batch size so we don't run into memory issues, instead of predicting on all frames. We are doing 20 at a time
        batch_size = 20
        detections = []
        #This will iterate through all the frames, and increment by the batch size (i=0, 20 ...)
        for i in range(0,len(frames), batch_size):
            #Now we will predict for the given batch size in the frame list, and using the 0.1 confidence
            #We used confidence = 0.1. This is the minimum and this is good enough to detect a lot of objects without many false detections
            #Instead of using self.model.track we will use the predict and overide the goalkeeper with the player and run a tracker on it after the detections using supervision
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            #Add the batch to the list of all the detections
            detections += detections_batch
        return detections
    
    #Function tries to also check for existing file in the stub path by reading it, otherwise runs the detections and tracking and returns objects
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        #Try to read from filesystem using pickle
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        #Dictionary of lists for players, refs and ball
        tracks={
            "players":[], # example of 1 frame's dictionary {0:{"bbox"}:[0,0,0,0]},1:{"bbox"}:[0,0,0,0]},2:{"bbox"}:[0,0,0,0]}} there can be additions/deletions after every frame
            "referees":[],
            "ball":[]
        }

        #Looping over the detection 1 by 1, and use the index of the list using frame_num
        for frame_num, detection in enumerate(detections):
            #Class mapped to somthing like this ${0;person, 1;goalkeeper ...}
            cls_names = detection.names
            #We want it to map to somthing like this ${person;0, ball:2 ...}
            #Switches k as value and value as key from the class name items list
            cls_names_inv = {v:k for k,v in cls_names.items()}

            #Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object (from id=1 to id=2 in output)
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    #Not hardcoding to number, rather using the classnames
                    #Will replace 1 with 2
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            #Track Objects
            #Will add tracker object to the detections
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            #Loop over each track
            for frame_detection in detection_with_tracks:
                #extract the bounding box at index 0, of 0 because its the first index in the detections output 
                bbox = frame_detection[0].tolist()
                #Extract the class id and track_id from the detections output
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            #For abll we dont want to loop over detection with tracks, just the supervision
            for frame_detection in detection_supervision:
                #extract the bounding box at index 0, of 0 because its the first index in the detections output 
                bbox = frame_detection[0].tolist()
                #Extract the class id and track_id from the detections output
                cls_id = frame_detection[3]

                #Just one ball so we can get it with [1]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        #Write the run to the filesystem using pickle
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)

        #A dictionary of lists of dictionaries
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        #put at the bottom of the bounding box at y2
        #We want center of circle to be center of bounding box
        y2 = int(bbox[3])

        #get center for ellipse
        x_center, _ = get_center_of_bbox(bbox)
        #for radius of ellipse
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            #Radius of the circle but we need to provide 2
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            #A bit at the end of the circle around the player will not be drawn but looks good and can play around
            #gameified look
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        #Draw rectangle for number under the player
        rectangle_width = 40
        rectangle_height = 20
        #Top left corner. Move have of the width from the center of the rectangle
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        #Check if there is a track id and draw a rectangle if thats the case
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            #Visual Features and adjustments for larger number displayed
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10


            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        return frame

    #Draw the triangles on the ball
    def draw_triangle(self, frame, bbox, color):
        #Bottom point of y, y1 because we want the triangle on top of the ball
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        #Define the three points of the triangle, numpy array of the 3 points
        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color,cv2.FILLED)
        #Draw a border, with same coordinates not filled but has edges
        cv2.drawContours(frame, [triangle_points],0,(0,0,0),2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        #Draw a semi transparent rectangle to display metrics
        #Overlay helps with transparency
        overlay = frame.copy()
        # cv2.rectangle(overlay, (1350,850), (1900,970), (255,255,255), -1)
        cv2.rectangle(overlay, (1350,850), (1900,970), (255,255,255), cv2.FILLED)
        #Transparency is 40%
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        #List until the frame number we are standing on
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        #Get the number of time each team has the ball, list must be numpy
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame,f"Team 1 Ball Control: {team_1*100:.2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame,f"Team 2 Ball Control: {team_2*100:.2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    #Function to create a circle instead of the existing boxes around the players
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        #List containing all the video frames with output drawn on them
        output_video_frames = []
        #Loop over each frame
        for frame_num, frame in enumerate(video_frames):
            #Start drawing process
            #Copy the frame to not pollute the frames coming in and the original list being passed is not being drawn on
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #Draw players ellipses
            for track_id, player in player_dict.items():
                #Color of the team
                color = player.get("team_color", (0,0,255))
                #Draw an ellipse with a red color
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                #Draw triangle over player that has ball currently
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,255,255))

            #Draw referee ellipses
            for _, referee in referee_dict.items():
                #Draw an ellipse with a red color
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))

            #Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball["bbox"],(0,255,0))

            #Draw Team Ball control percentage
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames