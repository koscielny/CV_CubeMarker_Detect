#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "CVHelper.hpp"
#include "CVMathExt.hpp"
#include "XMLParser.hpp"
#include "CVTypes.hpp"
#include "OppenEngine.hpp"
#include "TTagRecognizer.hpp"

#ifdef _WIN32
#include <iostream>
#include <opencv2/highgui.hpp>
#endif

    TTagRecognizer::TTagRecognizer() :
        length_threshold_(120.0f),
        area_threshold_(1000.0f) {
        template_tag_.Init();
    }

    TTagRecognizer::~TTagRecognizer() {

    }

    void TTagRecognizer::Init(const char *param) {
        intrinsics_.create(3, 3, CV_64FC1);
        intrinsics_ = Parser::Instance("..\\config.xml")->
            GetIntrinsics("Intrin_.Logitech480p.value");
        intrinsics_inv_ = intrinsics_.inv();
    }

    void TTagRecognizer::ProcessRGBFrame(const cv::Mat& input_image) {
        cv::Mat read_only_frame = input_image;
        CVCtorsVectorT contour_corners_detected;
        CVCtorsVectorT contour_all;
        CVCtorsHierarchyT contour_hierarchy;
        CVCtorsHierarchyT quads_hierarchy;
        CVCtorsVectorT quads_contours;

#ifdef _WIN32
        cv::Mat contour_display_frame(read_only_frame.size(), CV_8UC3, cv::Scalar(0));;
        cv::Mat mark_dispaly_frame = read_only_frame.clone();
        cv::Mat quad_display_frame(read_only_frame.size(), CV_8UC3, cv::Scalar(0));;
#endif // _WIN32

        cvtColor(read_only_frame, gray_img_, CV_BGR2GRAY);
        cv::adaptiveThreshold(gray_img_, binary_img_, 255.0,
            CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, kBinaryBlockSize, kBinaryOffset);
        // !cv::morphologyEx(binary_frame, binary_frame, MORPH_OPEN, Mat());

        // Contours
        // detect candidate before filter : increase the computational complexity,
        // While it's hard to maintain hierarchy when filtering
        cv::Mat temp_bin = binary_img_.clone(); //binary_img is still in need
        cv::findContours(temp_bin, contour_all, contour_hierarchy,
            CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

        // isQuad test, reduce hierarchy
        ContoursUtils::Instance()->
            FindQuadsInContours(contour_all, quads_contours, contour_hierarchy, quads_hierarchy);

        // detect boundary feature and find deeper candidate
        ContoursUtils::Instance()->
            FindMarkInQuads(quads_contours, contour_corners_detected, quads_hierarchy, binary_img_);

#ifdef _WIN32
        cv::drawContours(contour_display_frame, contour_all, -1, CV_RGB(255, 255, 0));
        cv::imshow("AllContour", contour_display_frame);
#endif
#ifdef _WIN32
        cv::drawContours(quad_display_frame, quads_contours, -1, CV_RGB(255, 255, 0));
        cv::imshow("AllQuad", quad_display_frame);
#endif
#ifdef _WIN32
        cv::Mat gray_img_C3;
        cv::cvtColor(gray_img_, gray_img_C3, CV_GRAY2RGB);
#endif

        cv::TermCriteria criteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 0.00001);
        cv::Size win_size(3, 3);
        cv::Size zero_zone(-1, -1);
        for (auto &refined_corners : contour_corners_detected) {
            cv::cornerSubPix(gray_img_, refined_corners, win_size, zero_zone, criteria);
            cv::Mat homography =
                cv::findHomography(template_tag_.get_template_tag_corners(), refined_corners);
            int32_t tag_code = template_tag_.Decode(refined_corners, gray_img_, homography);

#ifdef _WIN32
            cv::Mat warped_mark;
            cv::warpPerspective(mark_dispaly_frame, warped_mark, homography.inv(),
                cv::Size(template_tag_.get_template_tag_width(), template_tag_.get_template_tag_height()));
            cv::imshow("vis2", warped_mark);
#endif
            homography = intrinsics_inv_ * homography;
            float lambda0 = 1.0f / (float)cv::norm(homography.col(0));
            float lambda1 = 1.0f / (float)cv::norm(homography.col(1));
            auto r3 = homography.col(0).cross(homography.col(1));
            cv::normalize(r3, r3);
            cv::Mat rot = homography.clone();
            rot.col(2) = r3;
            OrthogonalizeRotationFast(rot, rot);

            cv::Vec3f t = cv::Mat(homography.col(2) * lambda0);
            RecognizedTag recognized_tag(tag_code);
            recognized_tag.found = true;
            recognized_tag.position.x = t[0];
            recognized_tag.position.y = t[1];
            recognized_tag.position.z = t[2];
            if (isinf(recognized_tag.position.x) || isinf(recognized_tag.position.y) || isinf(recognized_tag.position.z)) {
                continue;
            }
            // Type should be double
            recognized_tag.rotation = oe2::QuatFromRotation<double>(rot);
            oe2::OppenEngine::Instance()->SetRecognizedTag(recognized_tag);
#ifdef _WIN32
            cv::drawContours(mark_dispaly_frame, contour_corners_detected,
                &corners_i - &contour_corners_detected[0], CV_RGB(255, 255, 0));
            cv::drawContours(gray_img_C3, contour_corners_detected,
                &corners_i - &contour_corners_detected[0], CV_RGB(255, 0, 0));
            cv::circle(mark_dispaly_frame, refined_corners[0], 3, CV_RGB(0, 255, 0), 2);
            cv::circle(mark_dispaly_frame, refined_corners[1], 3, CV_RGB(0, 255, 255), 2);
            cv::putText(mark_dispaly_frame, std::to_string(tag_code),
                refined_corners[3], CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 50, 0));
#endif
        }
#ifdef _WIN32
        cv::imshow("Gray", gray_img_C3);
#endif
#ifdef _WIN32
        cv::imshow("vis", mark_dispaly_frame);
#endif
    }
