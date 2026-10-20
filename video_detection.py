
import cv2
import time
from pathlib import Path
from tkinter import Tk, filedialog
from ultralytics import YOLO


def run_video_detection(model):
    """Detect objects in a selected video and save the result."""

    # Select an input video
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        video_path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    if not video_path:
        print("No video selected.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open the selected video.")
        return

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    # Select output location
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        output_path = filedialog.asksaveasfilename(
            title="Save annotated video",
            defaultextension=".mp4",
            initialfile=f"{Path(video_path).stem}_detected.mp4",
            filetypes=[
                ("MP4 video", "*.mp4"),
                ("AVI video", "*.avi"),
            ],
        )
    finally:
        root.destroy()

    if not output_path:
        cap.release()
        print("Output location not selected. Video processing cancelled.")
        return

    # Configure video writer
    extension = Path(output_path).suffix.lower()

    if extension == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        writer.release()
        print("Error: Could not create the output video.")
        return

    frame_number = 0
    start_time = time.perf_counter()
    window_name = "Video Object Detection"

    print("\nVideo processing started.")
    print("Press Q to stop processing early.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            results = model(frame, verbose=False)
            result = results[0]
            annotated_frame = result.plot()

            object_count = (
                len(result.boxes)
                if result.boxes is not None
                else 0
            )

            elapsed = time.perf_counter() - start_time
            processing_fps = (
                (frame_number + 1) / elapsed
                if elapsed > 0
                else 0
            )

            cv2.putText(
                annotated_frame,
                f"Frame: {frame_number + 1}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Objects: {object_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Processing FPS: {processing_fps:.1f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            writer.write(annotated_frame)
            cv2.imshow(window_name, annotated_frame)

            frame_number += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Processing stopped by user.")
                break

            if (
                cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE,
                ) < 1
            ):
                print("Video window closed.")
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Frames processed: {frame_number}")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    model = YOLO("yolo26n.pt")
    run_video_detection(model)
