# Use this file to play around with YOLO using ultralytics
from ultralytics import YOLO

#load trained model
# model = YOLO('yolo8x')
model = YOLO('models/best.pt')

#run the model on the input video and save the output of the results of the prediction back to the results variable
results = model.predict('input_videos/08fd33_4.mp4',save=True)
#print results of the first frame
print(results[0])
print('---------------------------------------------')
for box in results[0].boxes:
    print(box)