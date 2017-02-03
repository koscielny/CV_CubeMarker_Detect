#ifndef TTAGRECOGNIZER_HPP
#define TTAGRECOGNIZER_HPP

#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <opencv2/core.hpp>
#include "ContoursUtils.hpp"
#include "TagTemplate.hpp"

class TTagRecognizer {
public:
        TTagRecognizer();
        virtual ~TTagRecognizer();

        virtual void Init(const char *param);
        virtual void ProcessRGBFrame(const cv::Mat& input_img);
    private:
        enum {
            kBinaryBlockSize = 7,
            kBinaryOffset = 3,
            kContourMethod = CV_CHAIN_APPROX_SIMPLE
        };

        cv::Mat gray_img_;
        cv::Mat binary_img_;

        std::vector<cv::Point2f> template_corners_;

        cv::Mat intrinsics_;
        cv::Mat intrinsics_inv_;

        float length_threshold_;
        float area_threshold_;
        TagTemplate template_tag_;
};

#endif