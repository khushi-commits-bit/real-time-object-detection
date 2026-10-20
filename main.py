
import cv2
import time
from collections import Counter
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit(1)

# Detection settings
confidence_threshold = 0.50

# COCO class IDs for common vehicles
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
selected_classes = None  # None means detect all classes

previous_time = time.perf_counter()
smoothed_fps = 0.0

print("Real-Time Object Detection Started!")
print("+ / - : Increase / decrease confidence")
print("1: All objects | 2: People | 3: Vehicles")
print("Q: Quit")

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Could not read webcam frame.")
        break

    # Run YOLO with the current settings
    inference_start = time.perf_counter()

    results = model(
        frame,
        conf=confidence_threshold,
        classes=selected_classes,
        verbose=False,
    )

    inference_time = (time.perf_counter() - inference_start) * 1000

    result = results[0]
    annotated_frame = result.plot()

    # Count detected objects by class
    class_counts = Counter()

    if result.boxes is not None and result.boxes.cls is not None:
        for class_id in result.boxes.cls.cpu().tolist():
            class_name = result.names[int(class_id)]
            class_counts[class_name] += 1

    total_objects = sum(class_counts.values())

    # Calculate smoothed FPS
    current_time = time.perf_counter()
    frame_time = current_time - previous_time
    previous_time = current_time

    if frame_time > 0:
        current_fps = 1.0 / frame_time

        if smoothed_fps == 0:
            smoothed_fps = current_fps
        else:
            smoothed_fps = (
                0.9 * smoothed_fps + 0.1 * current_fps
            )

    # Display status and performance information
    status = (
        "All objects"
        if selected_classes is None
        else "People only"
        if selected_classes == [0]
        else "Vehicles only"
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {smoothed_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Inference: {inference_time:.1f} ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Objects: {total_objects}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Confidence: {confidence_threshold:.2f}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Filter: {status}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    # Display per-class counts
    y_position = 180

    for class_name, count in sorted(class_counts.items()):
        cv2.putText(
            annotated_frame,
            f"{class_name}: {count}",
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        y_position += 25

    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Handle keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key in (ord("+"), ord("=")):
        confidence_threshold = min(
            0.95, round(confidence_threshold + 0.05, 2)
        )

    elif key in (ord("-"), ord("_")):
        confidence_threshold = max(
            0.05, round(confidence_threshold - 0.05, 2)
        )

    elif key == ord("1"):
        selected_classes = None

    elif key == ord("2"):
        selected_classes = [0]  # person

    elif key == ord("3"):
        selected_classes = vehicle_classes

cap.release()
cv2.destroyAllWindows()

print("Object detection stopped.")
