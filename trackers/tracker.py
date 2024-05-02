#Tracker to trace bounding box locations
from ultralytics import YOLO
import supervision as sv
import pickle
import os

class Tracker:
    #Gets called when we initialize the class
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
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


