#ifndef CONTOURSUTILS_HPP
#define CONTOURSUTILS_HPP

#include <opencv2\opencv.hpp>
#include "CVTypes.hpp"

    // Designed only for Contours Detect, sample usage:
    // ContoursUtils::Instance()->DetctQuad(...);
    // ContoursUtils::Instance()->DetectMark(...);
    class ContoursUtils {
    public:
        ~ContoursUtils();
        static ContoursUtils *Instance() {
            static ContoursUtils m_instance;
            return &m_instance;
        }

        // find quad (eps approx, 4 corners, enough area)
        // build quad hierarchy for detect deepest mark
        void FindQuadsInContours(const CVCtorsVectorT& src_contours,
            CVCtorsVectorT& output_quads_contours,
            const CVCtorsHierarchyT& src_hierarchy,
            CVCtorsHierarchyT &quads_tree_hierarchy);

        // bresenham test
        // get deepest mark candidates
        void FindMarkInQuads(CVCtorsVectorT& quads_contours,
            CVCtorsVectorT& detected_contours,
            const CVCtorsHierarchyT& quads_tree_hierarchy,
            cv::Mat& bin_img);

        // conversion tool between vector data
        // usually for vector<Point2i> to vector<Point2f>
        template <class SrcType, class DstType>
        void ConvertVectorTool(std::vector<SrcType>& src, std::vector<DstType>& dst) {
            dst.resize(src.size());
            std::copy(src.begin(), src.end(), dst.begin());
        }

        // void AddMarkers(std::vector<Marker>& possible_markers,
        //      const std::vector<std::vector<cv::Point2f> >& contour_points_detected);

    private:
        ContoursUtils();
        enum {
            KTheta = 1,
            KMinArea = 500
        };
        const float KApproxPolyEps = 0.15;
        const float KBresenRatio = 0.6;

        // supporting function for DetectQuad and DetectMark implementation
        void StackTraversal(const CVCtorsHierarchyT& src_hierarchy,
            CVCtorsHierarchyT& quads_tree_hierarchy,
            const std::vector <std::pair<bool, int> >& is_quad);
        bool IsQuad(CornersIntT contour,
            CornersIntT& candidate_quad);
        void FindDeeperCandidate(std::vector<std::pair<cv::Vec4i, bool>> & tree,
            int tree_iter, int pre_detected,
            const CVCtorsVectorT& src_contours,
            CVCtorsVectorT& detected_contours,
            std::vector<int>& candidate_queue);
        bool IsCandidateContour(CornersIntT& contour,
            cv::Mat& bin_img);
        void BresenhamBorder(cv::Point2f start_p,
            cv::Point2f end_p,
            int& good,
            int& dirty,
            cv::Mat& bin_img);
        void PixelTest(cv::Point2f& draw_pixel,
            cv::Point2f& right_theta,
            cv::Point2f& left_theta,
            int& good,
            int& dirty,
            cv::Mat& bin_img);
        void ComputeTheta(bool is_right,
            bool is_up,
            bool is_steep,
            cv::Point2f& right_theta,
            cv::Point2f& left_theta);
        bool IsWhitePixel(cv::Point2f pos,
            const cv::Mat& bin_img);
        bool IsImageBound(cv::Point2f pos,
            const cv::Mat& img);
    };

#endif