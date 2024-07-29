import cv2

# Open a connection to the webcam (usually the first webcam is index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Video stream opened successfully.")

# Define the codec and create VideoWriter object
# 'mp4v' is the codec for mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Make sure the dimensions are correct for your camera
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print(f"Frame width: {frame_width}, Frame height: {frame_height}")

out = cv2.VideoWriter('output30.mp4', fourcc, 30.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    

    # Write the frame into the file 'output.mp4'
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
