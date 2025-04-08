#ifndef YOLO_11_OBJECT_DETECTOR_HPP
#define YOLO_11_OBJECT_DETECTOR_HPP

#include "yolo_object_detector.hpp"


class Yolo11ObjectDetector : public YoloObjectDetector {
public:
    // ================================
    // Constants
    // ================================
    static constexpr int DEFAULT_IMAGE_SIZE = 640;

    // ================================
    // Functions
    // ================================
    Yolo11ObjectDetector(const cv::Size& imageDims = cv::Size(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
        : YoloObjectDetector(imageDims) {}

    std::vector<BoundingBox> ParseOutput(const float* output,
                                        int imageWidth,
                                        int imageHeight,
                                        float confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
                                        float iouThreshold = DEFAULT_IOU_THRESHOLD) override;
};

#endif
