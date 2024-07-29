import cv2
from ultralytics import YOLO
import numpy as np
import streamlit as st
# Load the YOLOv8 model
model = YOLO(r"models\model_for_tracking.pt") 


st.title("Well Vision")

show_image = st.image([])

# Open the video file
video_path = r"D:\Downloads\video.mp4"
# video_path = "http://192.168.137.198:8080/video_feed"
cap = cv2.VideoCapture(video_path)



# Make sure the dimensions are correct for your camera
frame_width = 500
frame_height = 500


print(f"Frame width: {frame_width}, Frame height: {frame_height}")


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame,(600,600))
        # frame = frame[100:600,200:450]
        # Run YOLOv8 inference on the frame
        results = model.predict(source=frame,iou=0.6,conf=0.35)

        # Visualize the results on the frame
        frame_plot = results[0].plot()

        final_result = np.concatenate([frame,frame_plot],axis=1)
        
        print(final_result.shape)
        cv2.putText(final_result,"Input frame",(10,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2) 
        cv2.putText(final_result,"Output frame",(260,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2) 
        
        
        # cv2.imshow("YOLOv8 Inference", final_result)
        show_image.image(final_result)  
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):  
            break

    
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()