#ifndef TAGTEMPLATE_HPP
#define TAGTEMPLATE_HPP

#include <vector>
#include <opencv2/core.hpp>
#include "IRecognizer.hpp"
#include "IIdentifiable.hpp"

    class TagTemplate {
    public:
        TagTemplate() {};
        void Init();

        int Decode(std::vector<cv::Point2f>& roi_corners, cv::Mat gray_img_, cv::Mat& homography);

        std::vector<cv::Point2f> get_template_tag_corners();
        float get_template_tag_width();
        float get_template_tag_height();
    private:
        const float kThreshConfidence = 1.4;
        std::vector<cv::Point2f> template_tag_corners_;
        std::vector<cv::Point2f> templete_code_samples_;
        std::vector<cv::Point2f> templete_align_samples_;

        float template_tag_width_;
        float template_tag_height_;
    };
#endif