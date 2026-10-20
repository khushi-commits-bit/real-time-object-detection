# Real-Time Object Detection Using YOLO and OpenCV

A computer vision project that detects and identifies objects in real time using a webcam, a pretrained YOLO model, and OpenCV.

## Overview

This project captures live video from a webcam, processes each frame using YOLO, and displays detected objects with bounding boxes and labels.

It demonstrates practical applications of computer vision and deep learning.

## Features

- Real-time object detection through a webcam
- Pretrained YOLO model for object recognition
- Bounding boxes and class labels for detected objects
- Live video display using OpenCV
- Simple keyboard control to stop detection

## Technologies Used

- **Python** — programming language
- **YOLO (Ultralytics)** — object detection model
- **OpenCV** — webcam access and video processing

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd real-time-object-detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this command in the same terminal first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

Ensure your webcam is connected and available. The YOLO model weights will be downloaded automatically when needed.

## How to Use

1. Run `main.py`.
2. Allow webcam access if prompted.
3. Point the webcam at objects you want to detect.
4. Press **Q** to close the detection window.

## Future Improvements

- Display object counts for each detected class
- Measure and display frames per second (FPS)
- Allow users to select different YOLO models
- Save detection results and video recordings
- Explore deployment on robotics and edge-computing platforms

## Learning Outcomes

- Practical use of pretrained deep learning models
- Real-time video processing with OpenCV
- Webcam integration using Python
- Project organization and dependency management

## License

This project is available for educational and portfolio purposes. A license can be added as the project develops.