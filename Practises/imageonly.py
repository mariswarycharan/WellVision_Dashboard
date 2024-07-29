
import cv2
from ultralytics import YOLO


# to load the yolo weights file using torch hub
model = YOLO(r"weights_file_for_silver_beads\silver_beads_yolov8_nano.pt") 

classes_list = model.names


def sort_array_func(val):
    return val[3]


# input image

frame = cv2.imread(r"D:\Downloads\WhatsApp Image 2023-10-13 at 2.58.38 PM.jpeg")
frame_without_condition = frame.copy()

results = model.predict(source=frame,iou=0.6,conf=0.25)
result_frame = results[0].plot()
cv2.imshow("YOLOv8 Inference",result_frame )