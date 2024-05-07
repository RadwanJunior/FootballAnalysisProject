# Import necessary libraries
import cv2
import sys

# Add the parent directory to the system path for module import
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator:
    def __init__(self):
        # Initialize frame window and frame rate for speed calculation
        self.frame_window = 5  # Calculate speed over every 5 frames
        self.frame_rate = 24   # Assuming a frame rate of 24 frames per second

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        # Loop through all tracked objects to calculate speed and distance
        for object, object_tracks in tracks.items():
            # Skip calculating for the ball and referees
            if object == "ball" or object == "referees":
                continue 

            # Determine the number of frames for the current object
            number_of_frames = len(object_tracks)

            # Loop through frames in steps of 'frame_window' to calculate distance and speed
            for frame_num in range(0, number_of_frames, self.frame_window):
                # Ensure not to exceed the bounds of the frame list
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                # Iterate through all tracks in the current frame
                for track_id,_ in object_tracks[frame_num].items():
                    # Skip if track does not exist in the last frame of this window
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Get positions at the start and end of the window. To only calculate objects that are in both frames
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # If positions are not recorded, skip to the next track
                    #Is none when the player is outside the trapezoidal shape
                    if start_position is None or end_position is None:
                        continue

                    # Calculate the distance covered between positions
                    distance_covered = measure_distance(start_position, end_position)
                    # Calculate time elapsed during this window per second
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    # Convert distance and time to speed in meters per second
                    speed_meteres_per_second = distance_covered / time_elapsed
                    # Convert speed to kilometers per hour
                    speed_km_per_hour = speed_meteres_per_second * 3.6

                    # Store total distance covered by each object/player
                    if object not in total_distance:
                        total_distance[object] = {}

                    #If there is no distance covered in this frame measured b4, add it
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    #Add to the total distance of the object covered in this frame
                    total_distance[object][track_id] += distance_covered

                    # Update the track info with calculated speed and total distance for all frames in the window
                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        # Loop through all frames to draw speed and distance information
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                # Skip drawing for ball and referees
                if object == "ball" or object == "referees":
                    continue
                # Draw speed and distance for each track on the frame
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed',None)
                        distance = track_info.get('distance',None)
                        if speed is None or distance is None:
                           continue

                        bbox = track_info['bbox']
                        # Calculate position for displaying text based on bounding box
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40  # Offset the text position vertically for visibility
                        position = tuple(map(int, position))

                        # Draw speed and distance on the frame
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Append modified frame to the output list
            output_frames.append(frame)
        
        return output_frames  # Return all frames with the drawn information
