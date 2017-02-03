#include "prefix.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "XMLParser.hpp"
#include "TagTemplate.hpp"

#ifdef _WIN32
#include <iostream>
#include <opencv2/highgui.hpp>
#endif

    void TagTemplate::Init() {
        template_tag_corners_.resize(4);
        template_tag_corners_ =
            Parser::Instance("../config.xml")->GetPoint2fArray("_TagTemplate.corners.value");

        templete_align_samples_.resize(4);
        templete_align_samples_ =
            Parser::Instance("../config.xml")->GetPoint2fArray("_TagTemplate.align.value");

        templete_code_samples_.resize(6);
        templete_code_samples_ =
            Parser::Instance("../config.xml")->GetPoint2fArray("_TagTemplate.code.value");

        template_tag_width_ = sqrtf((template_tag_corners_[0] - template_tag_corners_[1]).dot
        (template_tag_corners_[0] - template_tag_corners_[1]));
        template_tag_height_ = sqrtf((template_tag_corners_[1] - template_tag_corners_[2]).dot
        (template_tag_corners_[1] - template_tag_corners_[2]));
    }

    std::vector<cv::Point2f> TagTemplate::get_template_tag_corners() {
        return template_tag_corners_;
    }

    float TagTemplate::get_template_tag_width() {
        return template_tag_width_;
    }

    float TagTemplate::get_template_tag_height() {
        return template_tag_height_;
    }

    int TagTemplate::Decode(std::vector<cv::Point2f>& roi_corners, cv::Mat gray_img, cv::Mat& homography) {
        std::vector<cv::Point2f> roi_align_samples_;
        std::vector<cv::Point2f> roi_code_samples_;
        std::vector<cv::Point2f> roi_corners_aligned;
        roi_align_samples_.resize(4);
        roi_code_samples_.resize(6);
        cv::perspectiveTransform(templete_align_samples_, roi_align_samples_, homography);

        uchar max_density = 0;
        size_t base_corner = 0;
        for (auto &align_p : roi_align_samples_) {
            if (gray_img.at<uchar>(align_p) > max_density) {
                max_density = gray_img.at<uchar>(align_p);
                base_corner = &align_p - &roi_align_samples_[0];
            }
        }
        if (gray_img.at<uchar>(roi_align_samples_[base_corner % 4]) /
            gray_img.at<uchar>(roi_align_samples_[(base_corner + 1) % 4]) < kThreshConfidence)
            return -1; // weak judgement 
        uchar code_threshhold = (gray_img.at<uchar>(roi_align_samples_[base_corner % 4]) +
            gray_img.at<uchar>(roi_align_samples_[(base_corner + 1) % 4])) >> 1;
        if (gray_img.at<uchar>(roi_align_samples_[(base_corner + 2) % 4]) >= code_threshhold ||
            gray_img.at<uchar>(roi_align_samples_[(base_corner + 3) % 4]) >= code_threshhold)
            return -2; // weak judgement 

        roi_corners_aligned.resize(4);
        for (size_t i = 0; i < 4; i++)
            roi_corners_aligned[i] = roi_corners[(i + base_corner) % 4];

        cv::Mat trans_aligned =
            cv::findHomography(template_tag_corners_, roi_corners_aligned);
        cv::perspectiveTransform(templete_code_samples_, roi_code_samples_, trans_aligned);

        int code = 0;
        for (size_t i = 0; i < roi_code_samples_.size(); i++) {
            if (gray_img.at<uchar>(roi_code_samples_[i]) < code_threshhold) {
                code += (1 << i);
            }
        }
        roi_corners = roi_corners_aligned;
#ifdef _WIN32
#endif // WINDOWS
        return code;
    }
