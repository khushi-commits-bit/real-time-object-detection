
from ultralytics import YOLO

from webcam import run_webcam
from image_detection import run_image_detection
from video_detection import run_video_detection


def main():
    print("\n" + "=" * 45)
    print("       REAL-TIME OBJECT DETECTION")
    print("=" * 45)
    print("1. Webcam Detection")
    print("2. Image Detection")
    print("3. Video Detection")
    print("4. Exit")

    # Load the model once and reuse it
    model = YOLO("yolo26n.pt")

    while True:
        choice = input("\nSelect an option (1-4): ").strip()

        if choice == "1":
            run_webcam(model)

        elif choice == "2":
            run_image_detection(model)

        elif choice == "3":
            run_video_detection(model)

        elif choice == "4":
            print("Thank you for using the detector!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
