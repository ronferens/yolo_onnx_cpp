#include "yolo_11_object_detector.hpp"
#include <algorithm>

std::vector<BoundingBox> Yolo11ObjectDetector::ParseOutput(const float* output,
                                                        int imageWidth,
                                                        int imageHeight, 
                                                        float confidenceThreshold,
                                                        float iouThreshold) {
    constexpr int numClasses = 80; // Number of classes
    constexpr int numElements = 8400;  // Number of detection boxes

    std::vector<BoundingBox> detections;

    for (int i = 0; i < numElements; ++i) {
        float x = output[0 * numElements + i];  // channel 0
        float y = output[1 * numElements + i];  // channel 1
        float w = output[2 * numElements + i];  // channel 2
        float h = output[3 * numElements + i];  // channel 3

        // Find max class score
        float maxScore = -INFINITY;
        int classId = -1;

        for (int j = 0; j < numClasses; ++j) {
            float score = output[(4 + j) * numElements + i];
            if (score > maxScore) {
                maxScore = score;
                classId = j;
            }
        }

        float confidence = maxScore;

        if (confidence > confidenceThreshold) {
            // Apply letterbox transformation
            float imageX = (x - m_pad.x) / m_scale;
            float imageY = (y - m_pad.y) / m_scale;
            float imageW = w / m_scale;
            float imageH = h / m_scale;

            float xMin = imageX - (imageW / 2.0f);
            float yMin = imageY - (imageH / 2.0f);
            float xMax = imageX + (imageW / 2.0f);
            float yMax = imageY + (imageH / 2.0f);
            
            // Clamp to image bounds
            xMin = std::clamp(xMin, 0.0f, static_cast<float>(imageWidth - 1));
            yMin = std::clamp(yMin, 0.0f, static_cast<float>(imageHeight - 1));
            xMax = std::clamp(xMax, 0.0f, static_cast<float>(imageWidth - 1));
            yMax = std::clamp(yMax, 0.0f, static_cast<float>(imageHeight - 1));

            detections.push_back({static_cast<int>(xMin),
                                static_cast<int>(yMin),
                                static_cast<int>(xMax),
                                static_cast<int>(yMax),
                                confidence,
                                classId});
        }
    }

    // Apply Non-Maximum Suppression (NMS)
    return applyNMS(detections, iouThreshold);
} 