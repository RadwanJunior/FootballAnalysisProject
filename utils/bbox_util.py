#Function to get center of bbox
def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    #Get center and cast as integer
    return int((x1+x2)/2), int((y1+y2)/2)

#Get a bbox width
def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

#Measures distance between 2 points
def measure_distance(p1,p2):
    #The xs subtracted then power of 2, the ys subtracted then power of 2 then square root
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    #Return the distance between the two xs and two ys
    return p1[0]-p2[0], p1[1]-p2[1]

#Returns the postion of the foot of the player in the bounding box by getting the center of the x and just the y
def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)