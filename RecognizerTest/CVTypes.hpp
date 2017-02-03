#ifndef CVTYPES_HPP
#define CVTYPES_HPP

#include <array>
#include <opencv2/core.hpp>
typedef std::vector<std::vector<cv::Point2f> > CVCtorsVectorT;
typedef std::vector<cv::Vec4i> CVCtorsHierarchyT;
typedef std::vector<cv::Point2f> CornersFloatT;
typedef std::vector<cv::Point2i> CornersIntT;
//typedef std::array<cv::Point2f, 4> CornersFloatT;
//typedef std::array<cv::Point2i, 4> CornersIntT;

#endif