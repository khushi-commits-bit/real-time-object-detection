
import cv2
import time
from collections import Counter
from ultralytics import YOLO


def run_webcam(model):
    """Run real-time object detection using the webcam."""

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    confidence_threshold = 0.50
    vehicle_classes = [2, 3, 5, 7]
    selected_classes = None

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    print("\nWebcam detection started!")
    print("+ / - : Adjust confidence")
    print("1: All objects | 2: People | 3: Vehicles")
    print("Q: Quit webcam mode")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            start_time = time.perf_counter()

            results = model(
                frame,
                conf=confidence_threshold,
                classes=selected_classes,
                verbose=False,
            )

            inference_time = (
                time.perf_counter() - start_time
            ) * 1000

            result = results[0]
            annotated_frame = result.plot()

            # Count detected objects
            class_counts = Counter()

            if result.boxes is not None:
                for class_id in result.boxes.cls.cpu().tolist():
                    name = result.names[int(class_id)]
                    class_counts[name] += 1

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
                        0.9 * smoothed_fps
                        + 0.1 * current_fps
                    )

            # Current filter name
            if selected_classes is None:
                filter_name = "All objects"
            elif selected_classes == [0]:
                filter_name = "People only"
            else:
                filter_name = "Vehicles only"

            # Display performance information
            information = [
                f"FPS: {smoothed_fps:.1f}",
                f"Inference: {inference_time:.1f} ms",
                f"Objects: {total_objects}",
                f"Confidence: {confidence_threshold:.2f}",
                f"Filter: {filter_name}",
            ]

            y_position = 30

            for text in information:
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                y_position += 30

            # Display class counts
            for name, count in sorted(class_counts.items()):
                cv2.putText(
                    annotated_frame,
                    f"{name}: {count}",
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
                y_position += 25

            cv2.imshow(
                "Real-Time Object Detection",
                annotated_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key in (ord("+"), ord("=")):
                confidence_threshold = min(
                    0.95,
                    round(confidence_threshold + 0.05, 2),
                )

            elif key in (ord("-"), ord("_")):
                confidence_threshold = max(
                    0.05,
                    round(confidence_threshold - 0.05, 2),
                )

            elif key == ord("1"):
                selected_classes = None

            elif key == ord("2"):
                selected_classes = [0]

            elif key == ord("3"):
                selected_classes = vehicle_classes

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Webcam detection stopped.")
