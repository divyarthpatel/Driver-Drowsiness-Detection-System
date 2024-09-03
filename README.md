# Driver-Drowsiness-Detection-System
## Description

This project aims to develop a Machine Learning-based Drowsiness Detection System to enhance the safety of drivers by monitoring their eye movements and alerting them in case of drowsiness. The system utilizes computer vision and facial landmark detection techniques to identify signs of fatigue, such as eye closure, yawning, and head tilts, providing real-time alerts to prevent accidents caused by driver fatigue.

## Features

- **Real-Time Drowsiness Detection**: Continuously monitors the driver’s eyes and facial expressions to detect drowsiness.
- **Alert System**: Triggers audio alerts if drowsiness is detected to wake up the driver.
- **Head Pose Estimation**: Monitors head tilts to detect inattentiveness.
- **Eye Aspect Ratio Calculation**: Uses eye aspect ratio (EAR) to determine eye closure and blink duration.
- **Non-Intrusive**: Does not interfere with the driver’s vision or comfort while driving.
- **Works in Various Lighting Conditions**: Effective under varying lighting conditions including day and night.
- **Cross-Platform**: Compatible with multiple operating systems, including Windows, Linux, and macOS.

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - **OpenCV (cv2)**: For capturing and processing video frames.
  - **dlib**: For face detection and facial landmark extraction.
  - **NumPy**: For numerical operations.
  - **SciPy**: For scientific computations.
  - **Scikit-learn**: For machine learning algorithms and evaluations.
- **Tools**:
  - **Python IDE** (e.g., PyCharm, VS Code)
  - **Webcam**: For live video input.

## Prerequisites

Before running this project, ensure the following libraries are installed:

1. OpenCV (`cv2`)
2. dlib
3. NumPy
4. SciPy
5. Scikit-learn

### Installation

To install the required libraries, use the following command in your Windows Command Prompt:

```bash
pip install opencv-python dlib numpy scipy scikit-learn
