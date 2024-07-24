import streamlit as st
import cv2
from ultralytics import YOLO

# Set Streamlit title and description
st.title("Live Bearing Detection")
st.write("This is a live video feed from cameras with detection of bearings.")

# Load YOLO model
model = YOLO(r'wellvision2.pt')

# Create VideoCapture objects for two webcams
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  
cap3 = cv2.VideoCapture(2, cv2.CAP_DSHOW)  

conf = st.slider("Select the Confidence score", min_value=0, max_value=100, step=1)

col1, col2, col3 = st.columns(3)

with col1:
    image1 = st.image([])
    max_object_id1 = 0
    max_object_id_display1 = st.metric(label="Max Object ID Cam 1", value=max_object_id1)

with col2:
    image2 = st.image([])
    max_object_id2 = 0
    max_object_id_display2 = st.metric(label="Max Object ID Cam 2", value=max_object_id2)
with col3:
    image3 = st.image([])
    max_object_id3 = 0
    max_object_id_display3 = st.metric(label="Max Object ID Cam 3", value=max_object_id3)

# Check if the webcams are opened correctly
if not (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
    st.error("Error: Could not open all webcams.")
else:
    while cap1.isOpened() and cap2.isOpened() and cap3.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        if not (ret1 and ret2 and ret3):
            st.error("Error: Failed to capture image from one or more webcams.")
            break

        # Convert the frames from OpenCV's BGR format to RGB format
        rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        rgb_frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)


        # Perform object detection with YOLO on each frame
        result1 = model.track(frame1, conf=conf / 100)
        result2 = model.track(frame2, conf=conf / 100)
        result3 = model.track(frame3, conf=conf / 100)

        
        # Initialize max object IDs
        max_object_id1 = 0
        max_object_id2 = 0
        max_object_id3 = 0

        
        for res1 in result1:
            ids1 = res1.boxes.id
            if ids1==None:
                pass
            else:
                max_object_id1=max(ids1)
        
        for res2 in result2:
            ids2 = res2.boxes.id
            if ids2==None:
                pass
            else:
                max_object_id2=max(ids2)

        for res3 in result3:
            ids3 = res3.boxes.id
            if ids3==None:
                pass
            else:
                max_object_id3=max(ids3)
        

        # Update Streamlit metrics
        max_object_id_display1.metric(label="Max Object count 1", value=max_object_id1)
        max_object_id_display2.metric(label="Max Object count 2", value=max_object_id2)
        max_object_id_display3.metric(label="Max Object count 3", value=max_object_id3)

        # Display the annotated frames in Streamlit using st.image
        image1.image(result1[0].plot(), channels="BGR", use_column_width=True)
        image2.image(result2[0].plot(), channels="BGR", use_column_width=True)
        image3.image(result3[0].plot(), channels="BGR", use_column_width=True)


# Release the video capture objects
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()