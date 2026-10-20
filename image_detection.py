
import cv2
from pathlib import Path
from tkinter import Tk, filedialog
from ultralytics import YOLO


def run_image_detection(model):
    """Select an image, detect objects, and display results."""

    # Open a file picker
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        image_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    # Handle cancellation
    if not image_path:
        print("No image selected.")
        return

    # Read the selected image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not open the selected image.")
        return

    # Run YOLO detection
    results = model(image, verbose=False)
    result = results[0]

    # Draw bounding boxes and labels
    annotated_image = result.plot()

    # Count detected objects
    object_count = (
        len(result.boxes) if result.boxes is not None else 0
    )

    print(f"\nImage: {Path(image_path).name}")
    print(f"Objects detected: {object_count}")

    if result.boxes is not None:
        for class_id in result.boxes.cls.cpu().tolist():
            class_name = result.names[int(class_id)]
            print(f"Detected: {class_name}")

    # Display results
    cv2.imshow("Image Object Detection", annotated_image)
    print("Press Q or close the image window to finish.")

    while True:
        key = cv2.waitKey(100) & 0xFF

        if key == ord("q"):
            break

        # Also exit if the user closes the window
        if cv2.getWindowProperty(
            "Image Object Detection",
            cv2.WND_PROP_VISIBLE,
        ) < 1:
            break

    cv2.destroyAllWindows()

    # Ask whether to save the result
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        save_path = filedialog.asksaveasfilename(
            title="Save annotated image",
            defaultextension=".jpg",
            initialfile=f"{Path(image_path).stem}_detected.jpg",
            filetypes=[
                ("JPEG image", "*.jpg"),
                ("PNG image", "*.png"),
            ],
        )
    finally:
        root.destroy()

    if save_path:
        if cv2.imwrite(save_path, annotated_image):
            print(f"Annotated image saved to: {save_path}")
        else:
            print("Error: Could not save the annotated image.")
    else:
        print("Image result was not saved.")
