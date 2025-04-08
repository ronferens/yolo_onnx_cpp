#include "../include/yolo_object_detector.hpp"
#include <iostream>


YoloObjectDetector::YoloObjectDetector(const cv::Size& imageDims) : m_imageDims(imageDims)
{
    std::cout << "Initializing YOLO detector with image dimensions: " 
              << m_imageDims.width << "x" << m_imageDims.height << std::endl;
   
    // Configure the base class with these settings
    std::vector<const char*> output_node_names = { "output0" };
    std::vector<const char*> input_node_names = { "images" };
    std::vector<int64_t> input_dims = { 1, 3, m_imageDims.height, m_imageDims.width };

    SetInputNodeNames(&input_node_names);
    SetInputDemensions(input_dims);
    SetOutputNodeNames(&output_node_names);
}

YoloObjectDetector::~YoloObjectDetector() {
    std::cout << "Cleaning up YOLO detector resources" << std::endl;
}

cv::Mat YoloObjectDetector::LetterboxResize(const cv::Mat& image, const cv::Size& targetSize) {
    // Get image dimensions
    int imgH = image.rows;
    int imgW = image.cols;
    int targetH = targetSize.height;
    int targetW = targetSize.width;

    // Calculate scale to fit the image while maintaining aspect ratio
    m_scale = std::min(static_cast<float>(targetW) / imgW, static_cast<float>(targetH) / imgH);

    // Calculate new dimensions after scaling
    int newW = static_cast<int>(imgW * m_scale);
    int newH = static_cast<int>(imgH * m_scale);

    // Calculate padding to center the image
    m_pad.x = (targetW - newW) / 2;
    m_pad.y = (targetH - newH) / 2;

    // Resize image maintaining aspect ratio
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // Create letterboxed image with gray background (114, 114, 114)
    cv::Mat letterboxed = cv::Mat::zeros(targetH, targetW, image.type());
    letterboxed.setTo(cv::Scalar(114, 114, 114));
    
    // Copy resized image to the center of the letterboxed image
    resized.copyTo(letterboxed(cv::Rect(m_pad.x, m_pad.y, newW, newH)));

    return letterboxed;
}

bool YoloObjectDetector::PreprocessImage(const cv::Mat& image, std::vector<Ort::Value>& inputTensor)
{
    // Letterbox resize to 640x640
    m_letterboxedImage = LetterboxResize(image, cv::Size(m_imageDims.width, m_imageDims.height));
    
    // Convert the letterboxed image to blob
    m_blob = cv::dnn::blobFromImage(
        m_letterboxedImage,
        1/255.0,
        cv::Size(m_imageDims.width, m_imageDims.height),
        cv::Scalar(0, 0, 0),
        true,
        false,
        CV_32F);
    
    // Create a tensor from the blob
    try {
        inputTensor.emplace_back(Ort::Value::CreateTensor<float>(
                                m_memory_info,
                                m_blob.ptr<float>(),
                                m_blob.total(),
                                m_inputNodeDims.data(),
                                m_inputNodeDims.size()));
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl << ", Code: " << e.GetOrtErrorCode() << std::endl;
        return false;
    }

    return true;
}

bool YoloObjectDetector::DetectObjects(const cv::Mat& image, std::vector<Ort::Value>& outputTensor)
{
    // Preprocess the image
    std::vector<Ort::Value> inputTensor;
    bool success = PreprocessImage(image, inputTensor);
    if (!success)
    {
        return false;
    }

    // Run inference
    try {
        outputTensor = m_session.Run(Ort::RunOptions{ nullptr },
                                    m_inputNodeNames.data(),
                                    inputTensor.data(),
                                    inputTensor.size(),
                                    m_outputNodeNames.data(),
                                    1);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl << ", Code: " << e.GetOrtErrorCode() << std::endl;
        return false;
    }

    // Return the number of detected objects (elements in the output tensor)
    return (outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount() > 0);
}

float YoloObjectDetector::calculateIOU(const BoundingBox& box1, const BoundingBox& box2) {
    int x1 = std::max(box1.x_min, box2.x_min);
    int y1 = std::max(box1.y_min, box2.y_min);
    int x2 = std::min(box1.x_max, box2.x_max);
    int y2 = std::min(box1.y_max, box2.y_max);

    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int box1Area = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min);
    int box2Area = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min);

    return static_cast<float>(intersectionArea) / (box1Area + box2Area - intersectionArea);
}

std::vector<BoundingBox> YoloObjectDetector::applyNMS(std::vector<BoundingBox> detections, float iouThreshold) {
    std::vector<BoundingBox> filteredDetections;
    std::sort(detections.begin(), detections.end(), [](const BoundingBox& a, const BoundingBox& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        filteredDetections.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[i].class_id == detections[j].class_id && calculateIOU(detections[i], detections[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return filteredDetections;
}