# Yolo Object Detection using ONNX and C++

This repo demostrate how to run YOLO object detection model using ONNX model in C++.
The code was developed for Ubuntu 24.04 development environement.

## Requirements 

### Setup ONNX Runtime 	💻
1. Go to [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases).
2. Got the latest release, scroll down to the *Assets* section and download the onnxruntime-linux-x64-gpu-X.XX.X.tgz file.
3. Unzip the .zip file and place it in a selected folder. This will be our *ONNX_DIR*.
4. The unzipped folder should contain two folders: *include* and *lib*.

### Preparing a Yolo8/11 ONNX model 📂
1. Go to [Ultralytics website](https://docs.ultralytics.com/models/)
2. Select either [Yolo8](https://docs.ultralytics.com/models/yolov8/) or [Yolo11](https://docs.ultralytics.com/models/yolo11/) model to convert to ONNX.
3. Once you have downloaded the desired *.pt* file, you need to convert it to *.onnx* file. To do that follow the instruction in [this example](https://docs.ultralytics.com/integrations/onnx/#__tabbed_2_1).
4. Use the created *.onnx* model when running the code


## Running the Code 🚀
The code entry point is the *src/main_run_yolo.cpp* file
The required arguments are:
* ***use_cuda***: Use CUDA (true or false)
* ***model_path***: Path to the ONNX model
* ***video_path***: Path to the video file or camera index (0 for default camera)

---

## YOLOv8/11 ONNX Output Structure and Parsing Guide

### Output Tensor Shape

The output tensor returned from the model has the following dimensions:

```
[1, 84, 8400]
```

- `1`: Batch size (only one image at a time)
- `84`: Number of output channels per prediction
  - Channels `[0]`: x (center x coordinate)
  - Channels `[1]`: y (center y coordinate)
  - Channels `[2]`: w (width)
  - Channels `[3]`: h (height)
  - Channels `[4...83]`: class logits (80 classes)
- `8400`: Number of detection points (grid positions × anchors)

---

### Output Layout Schematic

The following schematic illustrates the structure of the flat float array:

```
output[0 * 8400 + i] → x
output[1 * 8400 + i] → y
output[2 * 8400 + i] → w
output[3 * 8400 + i] → h
output[4 * 8400 + i] → class_0 score
...
output[83 * 8400 + i] → class_79 score
```

Where:
- `i` ranges from `0` to `8399`
- Each block of 8400 values corresponds to one channel across all predictions

---

### Input Format (C++ Perspective)

The tensor is returned as a contiguous buffer of type `float*`, containing:

```
84 channels × 8400 elements = 705600 float values
```

The layout is row-major, so to access the score for a specific element `i` and channel `c`:

```cpp
float value = output[c * numElements + i];
```

Where:
- `output` is a `float*` pointing to the start of the data
- `numElements = 8400`
- `c` is the channel index (0 to 83)
- `i` is the prediction index (0 to 8399)

---

### Bounding Box and Class Parsing

To parse each detection:

1. Extract bounding box parameters:
   ```cpp
   float x = output[0 * numElements + i];
   float y = output[1 * numElements + i];
   float w = output[2 * numElements + i];
   float h = output[3 * numElements + i];
   ```

2. Loop through class logits and find the class with the highest score:
   ```cpp
   float maxScore = -INFINITY;
   int classId = -1;
   for (int j = 0; j < 80; ++j) {
       float score = output[(4 + j) * numElements + i];
       if (score > maxScore) {
           maxScore = score;
           classId = j;
       }
   }
   ```

3. Use the `maxScore` as confidence (optionally apply sigmoid if your model outputs logits):
   ```cpp
   float confidence = maxScore; // or sigmoid(maxScore)
   ```

4. Apply a threshold (e.g. 0.25) and store the detection:
   ```cpp
   if (confidence > 0.25f) {
       // Store detection with (x, y, w, h, confidence, classId)
   }
   ```

---

### Summary
- Output shape: `[1, 84, 8400]`
- Each detection point: 4 box values + 80 class scores
- Flat float buffer: size = 705600
- Access format: `output[channel * 8400 + index]`

This structure is efficient and standard for modern YOLO implementations exported to ONNX.


