#Tracker to trace bounding box locations
from ultralytics import YOLO
import supervision as sv

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

    def get_object_tracks(self,frames):
        detections = self.detect_frames(frames)

        #Looping over the detection 1 by 1, and use the index of the list using frame_num
        for frame_num, detection in enumerate(detections):
            #Class mapped to somthing like this ${0;person, 1;goalkeeper ...}
            cls_names = detections.names
            #We want it to map to somthing like this ${person;0, ball:2 ...}
            #Switches k as value and value as key from the class name items list
            cls_names_inv = {v:k for k,v in cls_names.items()}

            #Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            

