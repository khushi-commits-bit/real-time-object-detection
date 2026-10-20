
import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started! Press 'q' to quit.")

while True:
    # Read one frame from the webcam
    success, frame = cap.read()

    if not success:
        print("Error: Could not read webcam frame.")
        break

    # Detect objects in the current frame
    results = model(frame, verbose=False)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the video
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()