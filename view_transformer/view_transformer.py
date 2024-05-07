import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        #Field dimensions
        field_width = 68
        field_length = 23.32

        #Trapezoid using estimated values on field
        self.pixel_vertices = np.array([
            [110,1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])
        
        #Rectangle that we want the trapezoid to be after we transform it
        self.target_vertices = np.array([
            [0, field_width],
            [0,0],
            [field_length,0],
            [field_length,field_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

        #Transform adjusted points to the points after the transformation of the perspective transformation

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        #Check if point is inside traperzoid, if not we ignore speed of it
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0
        if not is_inside:
            return None
        
        #Just reshaping, won't do much to the numbers
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)

        return transform_point.reshape(-1,2)
    
    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
