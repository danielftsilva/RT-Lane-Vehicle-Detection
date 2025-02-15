<h1>Real-Time Lane and Vehicle Detection Using OpenCV and OpenCL</h1>

## Description
This project, titled "Real-Time Lane and Vehicle Detection Using OpenCV and OpenCL", was developed as part of the Advanced Topics in Digital Image Processing course. The objective is to create an application that processes a video recorded from a camera placed inside a car, detects lanes and vehicles, and marks vehicles in the same lane in red and others in green. The project leverages GPU for image processing, utilizing the Hough transform for lane detection and Cascade Classifiers for vehicle detection.

<h2>👨‍💻 Project Summary:</h2>

- **Video Input**: The system processes a video recorded from a camera inside a car.
- **Lane Detection**: Using the Hough transform implemented in OpenCL, the system detects lane lines and marks them with blue lines.
- **Vehicle Detection**: The system uses Cascade Classifiers from the OpenCV library to detect vehicles. Vehicles in the same lane are marked with red rectangles, while others are marked with green rectangles.
- **GPU Utilization**: The project makes extensive use of GPU for image processing tasks to enhance performance.

<h2>🛠️ Implementation Details:</h2>

- **Sobel Filter**: Applied to detect edges in the video frames.
- **Hough Transform**: Implemented in OpenCL to detect lane lines.
- **Cascade Classifiers**: Used for vehicle detection, with pre-trained classifiers for cars.
- **Visualization**: The processed video frames are displayed with marked lanes and vehicles.

<h2>📸 Results:</h2>

Below are some screenshots from the processed video (`video1.MTS`), showcasing the detection of lanes and vehicles:

<!-- Add your screenshots here -->

