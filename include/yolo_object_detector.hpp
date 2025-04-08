#ifndef YOLO_OBJECT_DETECTOR_HPP
#define YOLO_OBJECT_DETECTOR_HPP

#include "onnxinferencebase.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct BoundingBox {
    int x_min, y_min, x_max, y_max;
    float confidence;
    int class_id;
};

/**
 * @brief Base class for YOLO object detection implementations
 * 
 * This class provides common functionality for YOLO-based object detectors:
 * - Image preprocessing including letterboxing and normalization
 * - ONNX model inference
 * - Non-maximum suppression (NMS) for filtering overlapping detections
 * 
 * Derived classes must implement ParseOutput() to handle model-specific output formats.
 * 
 * The detector supports:
 * - Configurable input image dimensions (default 640x640)
 * - Adjustable confidence threshold for filtering weak detections
 * - Adjustable IOU threshold for NMS filtering
 * - CUDA acceleration when available
 * 
 * @see BoundingBox Struct containing detection results (coordinates, confidence, class)
 * @see OnnxInferenceBase Base class providing ONNX Runtime functionality
 */

class YoloObjectDetector : public OnnxInferenceBase {
    public:
        // ================================
        // Constants
        // ================================
        static constexpr int DEFAULT_IMAGE_SIZE = 640;
        static constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.5f;
        static constexpr float DEFAULT_IOU_THRESHOLD = 0.45f;

        // ================================
        // Functions
        // ================================
        YoloObjectDetector(const cv::Size& imageDims = cv::Size(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE));

        ~YoloObjectDetector();

        bool PreprocessImage(const cv::Mat& image, std::vector<Ort::Value>& inputTensor);

        bool DetectObjects(const cv::Mat& image, std::vector<Ort::Value>& outputTensor);

        virtual std::vector<BoundingBox> ParseOutput(const float* output,
                                                    int imageWidth,
                                                    int imageHeight,
                                                    float confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
                                                    float iouThreshold = DEFAULT_IOU_THRESHOLD) = 0;

    protected:
        // ================================
        // Functions
        // ================================
        std::vector<BoundingBox> applyNMS(std::vector<BoundingBox> detections, float iouThreshold);

        float calculateIOU(const BoundingBox& box1, const BoundingBox& box2);

        cv::Mat LetterboxResize(const cv::Mat& image, const cv::Size& targetSize);

        // ================================
        // Variables
        // ================================
        cv::Size m_imageDims;
        cv::Mat m_blob;
        cv::Mat m_letterboxedImage;
        float m_scale;
        cv::Point m_pad;
};

#endif
