#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include "yolo_11_object_detector.hpp"
#include "coco_classes.hpp"



void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <use_cuda> <model_path> <video_path>" << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  use_cuda  : Use CUDA (true or false)" << std::endl;
    std::cout << "  model_path  : Path to the ONNX model" << std::endl;
    std::cout << "  video_path  : Path to the video file or camera index (0 for default camera)" << std::endl;
}

int main(int argc, char** argv)
{
    // Check command line arguments
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "Initializing YOLO detector with:" << std::endl;
    std::cout << "  Use CUDA: " << argv[1] << std::endl;
    std::cout << "  Model path: " << argv[2] << std::endl;
    std::cout << "  Video path: " << argv[3] << std::endl;

    // Initialize YOLO detector
    Yolo11ObjectDetector yolo_model;
    yolo_model.ConfigureSession(atoi(argv[1]));
    yolo_model.LoadModel(argv[2]);
    
    int numObjectsDetected = 0;
    std::vector<Ort::Value> outputTensor;

    // Open video capture
    cv::VideoCapture cap;
    bool isCamera = isdigit(argv[3][0]);
    
    if (isCamera) {
        // If the argument is a number, treat it as a camera index
        int cameraIndex = atoi(argv[3]);
        std::cout << "Opening camera " << cameraIndex << std::endl;
        cap.open(cameraIndex);
    } else {
        // Otherwise treat it as a video file path
        std::cout << "Opening video file: " << argv[3] << std::endl;
        cap.open(argv[3]);
    }
    
    // Check if camera opened successfully
    if(!cap.isOpened()) {
        std::cerr << "Error opening video stream or file: " << argv[3] << std::endl;
        return -1;
    }

    // Get and print video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Video properties:" << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;
    if (fps > 0) {
        std::cout << "  FPS: " << fps << std::endl;
    }

    // Create window for display
    cv::namedWindow("Video Stream", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    int frameCount = 0;
    
    // Variables for FPS calculation
    auto startTime = std::chrono::high_resolution_clock::now();
    int framesSinceLastFPS = 0;
    float currentFPS = 0.0f;
    bool success = false;

    std::vector<BoundingBox> finalDetections;
    
    while(1) {
        // Capture new frame
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        // Increment frame count and frames since last FPS calculation
        frameCount++;
        framesSinceLastFPS++;
        
        // Calculate FPS every second
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0f;
        
        if (elapsedTime >= 1.0f) {
            currentFPS = framesSinceLastFPS / elapsedTime;
            framesSinceLastFPS = 0;
            startTime = currentTime;
        }

        // Run object detection
        success = yolo_model.DetectObjects(frame, outputTensor);
        if (success) {
            numObjectsDetected = 0;

            // Get the output tensor data
            float* Result = outputTensor.front().GetTensorMutableData<float>();

            finalDetections = yolo_model.ParseOutput(Result, width, height);
              
            // Drawing the boxes from finalDetections
            for (const auto& detection : finalDetections) {
                // Draw the bounding box
                cv::rectangle(frame, 
                            cv::Rect(detection.x_min, detection.y_min, 
                                   detection.x_max - detection.x_min, 
                                   detection.y_max - detection.y_min),
                            cv::Scalar(255, 255, 255), 1);
                
                // Add class label and confidence
                std::string label = coco::CLASS_NAMES[detection.class_id] + 
                                  "(" + std::to_string(detection.class_id) + ")" + ": " + 
                                  std::to_string(static_cast<int>(detection.confidence * 100)) + "%";

                cv::putText(frame, label, 
                          cv::Point(detection.x_min, detection.y_min - 10),
                          cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                numObjectsDetected++;
            }
            outputTensor.clear();

            std::cout << "Detected " << numObjectsDetected << " objects" << std::endl;
        }

        // Display FPS on frame
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(currentFPS));
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        // Display the frame
        cv::imshow("Video Stream", frame);

        // Break loop on 'ESC' or 'q' key press
        char c = (char)cv::waitKey(1);
        if(c == 27 || c == 'q' || c == 'Q') {
            break;
        }
    }

    std::cout << "Total frames processed: " << frameCount << std::endl;

    // Release the capture and destroy windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}