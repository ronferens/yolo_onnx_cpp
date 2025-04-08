# Yolo Object Detection using ONNX and C++

This repo demostrate how to run YOLO object detection model using ONNX model in C++.
The code was developed for Ubuntu 24.04 development environement.

## Requirements

### Setup ONNX Runtime
1. Go to [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases).
2. Got the latest release, scroll down to the *Assets* section and download the onnxruntime-linux-x64-gpu-X.XX.X.tgz file.
3. Unzip the .zip file and place it in a selected folder. This will be our *ONNX_DIR*.
4. The unzipped folder should contain two folders: *include* and *lib*.

### Preparing a Yolo8/11 ONNX model
1. Go to [Ultralytics website](https://docs.ultralytics.com/models/)
2. Select either [Yolo8](https://docs.ultralytics.com/models/yolov8/) or [Yolo11](https://docs.ultralytics.com/models/yolo11/) model to convert to ONNX.
3. Once you have downloaded the desired *.pt* file, you need to convert it to *.onnx* file. To do that follow the instruction in [this example](https://docs.ultralytics.com/integrations/onnx/#__tabbed_2_1).
4. Use the created *.onnx* model when running the code


## Running the Code
The code entry point is the *src/main_run_yolo.cpp* file
The required arguments are:
* use_cuda: Use CUDA (true or false)
* model_path: Path to the ONNX model
* video_path: Path to the video file or camera index (0 for default camera)
