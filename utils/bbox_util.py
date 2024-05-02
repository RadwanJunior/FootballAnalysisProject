#Function to get center of bbox
def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    #Get center and cast as integer
    return int((x1+x2)/2), int((y1+y2)/2)

#Get a bbox width
def get_bbox_width(bbox):
    return[2]-bbox[0]