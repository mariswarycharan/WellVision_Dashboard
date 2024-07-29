from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO(r"models\model_for_tracking.pt")

def point_inside(rectangle, entire_coordinates):    
    x1, y1 ,x2, y2 , _ , _ = rectangle
    return (entire_coordinates[0] <= x1 <= entire_coordinates[2] and entire_coordinates[1] <= y1 <= entire_coordinates[3]) or (entire_coordinates[0] <= x2 <= entire_coordinates[2] and entire_coordinates[1] <= y2 <= entire_coordinates[3])


# Open the video file
video_path = r"D:\Downloads\Defected rollers high speed.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

dist_for_decision = {}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame,tracker="botsort.yaml",classes=[0], persist=True)

        annotated_frame = results[0].plot()
        # # Get the boxes and track IDs
        allboxes = results[0].boxes.numpy().data.tolist()
        
        track_ids_check = results[0].boxes.id
        
        if track_ids_check is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            
            all_bearing_boxes = list(filter(lambda x: x[-1] == 1.0, allboxes))
            all_defect_boxes = list(filter(lambda x: x[-1] != 1.0, allboxes))
            
            
            for id,i in enumerate(all_bearing_boxes):
                sub_dist = {}
                for j in all_defect_boxes:
                    res_inside = point_inside(j,i)
                    # if res_inside :
                        
                
            print(track_ids)
            
            
            # Visualize the results on the frame

            # Plot the tracks
            # for box, track_id in zip(boxes, track_ids):
            #     x, y, w, h = box
            #     track = track_history[track_id]
            #     track.append((float(x), float(y)))  # x, y center point
            #     if len(track) > 30:  # retain 90 tracks for 90 frames
            #         track.pop(0)

            #     # Draw the tracking lines
            #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()