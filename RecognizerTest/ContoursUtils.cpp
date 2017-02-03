#include "prefix.hpp"
#include <stack>
#include "math.h"

#include "ContoursUtils.hpp"

namespace oe2 {

    void ContoursUtils::FindQuadsInContours(const CVCtorsVectorT& src_contours,
        CVCtorsVectorT& output_quads_contours,
        const CVCtorsHierarchyT& src_hierarchy,
        CVCtorsHierarchyT &quads_tree_hierarchy) {
        // hierarchy[i].first[0] , hiearchy[i][1] , hiearchy[i][2] , and hiearchy[i][3]
        // the next contour
        // the previous contour at the same hierarchical level
        // the first child contour
        // the parent contour
        // bool : is_candidate

        std::vector <std::pair<bool, int> > is_quad;
        CVCtorsVectorT quads_contours;

        is_quad.resize(src_contours.size());
        quads_contours.resize(src_contours.size());

        for (int i = 0; i < src_contours.size(); i++) {
            if (IsQuad(src_contours[i], quads_contours[i])) {
                is_quad[i].first = true;
                output_quads_contours.push_back(quads_contours[i]);
                is_quad[i].second = output_quads_contours.size() - 1;
            }
            else {
                is_quad[i].first = false;
            }
        }//is_quad already , quads_contours already , output_quads_contours already

        quads_tree_hierarchy.resize(output_quads_contours.size(), cv::Vec4i(-1, -1, -1, -1));
        StackTraversal(src_hierarchy, quads_tree_hierarchy, is_quad);
    }

    void ContoursUtils::FindMarkInQuads(CVCtorsVectorT& quads_contours,
        CVCtorsVectorT& detected_contours,
        const CVCtorsHierarchyT& quads_tree_hierarchy,
        cv::Mat& bin_img) {
        if (quads_contours.size() > 0) {
            std::vector<std::pair<cv::Vec4i, bool>> CandidateTree;

            for (int i = 0; i < quads_contours.size(); i++) {
                CandidateTree.push_back(std::make_pair(quads_tree_hierarchy[i], false));
                if (IsCandidateContour(quads_contours[i], bin_img)) {
                    CandidateTree[i].second = true;
                    // std::cout << quads_tree_hierarchy[i] << std::endl;
                }
                //hierarchy failed
            }
            std::vector<int> candidate_queue;
            int root_iter = 0;
            while (root_iter >= 0) {
                //root_iter = src_hierarchy[root_iter][0];
                FindDeeperCandidate(CandidateTree, root_iter, -1,
                    quads_contours, detected_contours,
                    candidate_queue);// -1和cur_detected判定有关
                root_iter = quads_tree_hierarchy[root_iter][0];//next
            }
        }
    }

    void ContoursUtils::StackTraversal(const CVCtorsHierarchyT& src_hierarchy,
        CVCtorsHierarchyT& quads_tree_hierarchy,
        const std::vector <std::pair<bool, int> >& is_quad) {
        std::vector<int> root_chain;

        /* pair(current, parent)*/
        std::stack<std::pair<int, int>> src_tree_stack;
        int next_bro = 0;
        int root_parent = -1;

        while (next_bro >= 0) {
            src_tree_stack.push({ next_bro, root_parent });
            next_bro = src_hierarchy[next_bro][0];
        }

        while (!src_tree_stack.empty()) {//main loop
            std::pair<int, int> cur_iter = src_tree_stack.top();
            int cur_node = cur_iter.first;
            int cur_parent = cur_iter.second;
            int parent_of_kids;
            src_tree_stack.pop();

            if (is_quad[cur_node].first == true) {/*is quad node*/
                if (cur_parent == -1) {
                    root_chain.push_back(cur_node);
                }
                else {
                    //std::cout << "parent";
                    quads_tree_hierarchy[is_quad[cur_node].second][3] =
                        is_quad[cur_parent].second;//parent
                    int last_kid = quads_tree_hierarchy[is_quad[cur_parent].second][2];//first
                    if (last_kid == -1) {//no former child
                        quads_tree_hierarchy[is_quad[cur_parent].second][2] =
                            is_quad[cur_node].second;//firstchild
                    }
                    else {//already have kids
                        while (quads_tree_hierarchy[last_kid][0] >= 0) {
                            last_kid = quads_tree_hierarchy[last_kid][0];
                        }//go to last kid
                        quads_tree_hierarchy[is_quad[cur_node].second][1] = last_kid;
                        quads_tree_hierarchy[last_kid][0] = is_quad[cur_node].second;
                    }
                }
                parent_of_kids = cur_node;//set to cur_node
            }
            else {/*not quad node*/
                parent_of_kids = cur_parent;//set to cur_node's parent
            }

            int next_kid = src_hierarchy[cur_node][2];
            while (next_kid >= 0) {
                src_tree_stack.push({ next_kid, parent_of_kids });
                next_kid = src_hierarchy[next_kid][0];//kid's bro
            }
        }//end stack while

        if (0 < quads_tree_hierarchy.size()) {
            for (int i = 0; i < root_chain.size() - 1; i++) {
                quads_tree_hierarchy[is_quad[root_chain[i]].second][1] =
                    is_quad[root_chain[i + 1]].second;

                quads_tree_hierarchy[is_quad[root_chain[i + 1]].second][0] =
                    is_quad[root_chain[i]].second;
            }

            quads_tree_hierarchy[is_quad[root_chain[root_chain.size() - 1]].second][1] = -1;
            quads_tree_hierarchy[is_quad[root_chain[0]].second][0] = -1;
        }
    }

    bool ContoursUtils::IsQuad(CornersIntT contour,
        CornersIntT& detected_quad) {
        CornersIntT approx_poly;
        double eps = contour.size() * KApproxPolyEps;//eps的设置
        approxPolyDP(contour, approx_poly, eps, true);//
                                                      //凸四边形
        if (contourArea(contour) > KMinArea) {
            if (approx_poly.size() == 4 && isContourConvex(approx_poly)) {
                //Sort the points in anti-clockwise
                detected_quad = approx_poly;
                cv::Point v1 = detected_quad[1] - detected_quad[0];
                cv::Point v2 = detected_quad[2] - detected_quad[0];
                if (v1.cross(v2) > 0)    //由于图像坐标的Y轴向下，所以大于零代表逆时针
                    swap(detected_quad[1], detected_quad[3]);

                return true;
            }
        }
        return false;
    }

    void ContoursUtils::FindDeeperCandidate(std::vector<std::pair<cv::Vec4i, bool>> & tree,
        int tree_iter, int pre_detected,
        const CVCtorsVectorT& src_contours,
        CVCtorsVectorT& detected_contours,
        std::vector<int>& candidate_queue) {
        int cur_detected;
        tree[tree_iter].second == true ? cur_detected = tree_iter : cur_detected = pre_detected;

        int next_iter = tree[tree_iter].first[2];//first child
        if (next_iter >= 0) {//not deepest
            while (next_iter >= 0) {//if exist , >0
                FindDeeperCandidate(tree, next_iter, cur_detected,
                    src_contours, detected_contours, candidate_queue);
                next_iter = tree[next_iter].first[0];//next bro
            }
        }
        else {// already deepest
              // include candidate
            if (cur_detected < 0)
                return;

            // avoid repeating
            if (find(candidate_queue.begin(), candidate_queue.end(), cur_detected) ==
                candidate_queue.end()) {
                candidate_queue.push_back(cur_detected);
                detected_contours.push_back(src_contours[cur_detected]);
            }

        }
    }

    bool ContoursUtils::IsCandidateContour(CornersIntT& contour, cv::Mat& bin_img) {
        //Bresenham:for(every_side_in_poly)

        int good = 0, dirty = 0;
        //convert into counter clockwise
        std::vector<cv::Point> clock_wise_contour;
        cv::convexHull(contour, clock_wise_contour, true); // points sequence order changed
        for (int i = 0; i < contour.size(); i++) {
            //统计下good/dirty比例//i-1 到 i 逆时针，
            BresenhamBorder(contour[i], contour[(i + 1) % contour.size()], good, dirty, bin_img);
        }

        if (((float)(good + 1) / (float)(good + dirty + 1)) > KBresenRatio &&
            ((good + dirty) > 0)) {
            return true;
        }
        else {
            return false;
        }
    }

    void ContoursUtils::BresenhamBorder(cv::Point2f start_p, cv::Point2f end_p,
        int& good, int& dirty, cv::Mat& bin_img) {
        int x0 = start_p.x,
            y0 = start_p.y,
            x1 = end_p.x,
            y1 = end_p.y;
        cv::Point2f draw_pixel = start_p;
        int dx,             // difference in x's
            dy,             // difference in y's
            dx2,            // dx,dy * 2
            dy2,
            x_inc,          // amount in pixel space to move during drawing
            y_inc,          // amount in pixel space to move during drawing
            error,          // the discriminant i.e. error i.e. decision variable
            index;          // only used for looping

        // vector directing to right side, to compute right-side / left-side point
        cv::Point2f right_theta;
        cv::Point2f left_theta;
        dx = x1 - x0;
        dy = y1 - y0;
        // test which direction the line is going in i.e. slope angle
        if (dx >= 0) {
            x_inc = 1;
        }
        else {
            x_inc = -1;
            dx = -dx;  // absolute value
        }
        if (dy >= 0) {
            y_inc = 1;
        }
        else {
            y_inc = -1;
            dy = -dy;  // absolute value
        }
        dx2 = dx << 1;
        dy2 = dy << 1;

        ComputeTheta((x_inc > 0), (y_inc > 0), (dx <= dy), right_theta, left_theta);

        // now based on which delta is greater we can draw the line
        if (dx > dy) {
            error = dy2 - dx;
            for (index = 0; index <= dx; index++) {
                if (error >= 0) {
                    error -= dx2;
                    draw_pixel.y += y_inc;
                }
                error += dy2;
                draw_pixel.x += x_inc;
                PixelTest(draw_pixel, right_theta, left_theta, good, dirty, bin_img);
            }
        }
        else {
            error = dx2 - dy;
            for (index = 0; index <= dy; index++) {
                if (error >= 0) {
                    error -= dy2;
                    draw_pixel.x += x_inc;
                }
                error += dx2;
                draw_pixel.y += y_inc;
                PixelTest(draw_pixel, right_theta, left_theta, good, dirty, bin_img);
            }
        }

    }

    void ContoursUtils::PixelTest(cv::Point2f& draw_pixel,
        cv::Point2f& right_theta, cv::Point2f& left_theta,
        int& good, int& dirty,
        cv::Mat& bin_img) {
        if (!IsImageBound(draw_pixel + right_theta, bin_img) ||
            !IsImageBound(draw_pixel + left_theta, bin_img))
            return;

        if (IsWhitePixel(draw_pixel + right_theta, bin_img) &&
            !IsWhitePixel(draw_pixel + left_theta, bin_img))
            good++;
        else
            dirty++;
    }

    //isright or isleft
    void ContoursUtils::ComputeTheta(bool is_right, bool is_up, bool is_steep,
        cv::Point2f& right_theta, cv::Point2f& left_theta) {
        if (is_steep) {
            if (is_up) {
                right_theta = cv::Point2f(KTheta, 0);
                left_theta = cv::Point2f(-KTheta, 0);
            }
            else { // !isup
                right_theta = cv::Point2f(-KTheta, 0);
                left_theta = cv::Point2f(KTheta, 0);
            }
        }
        else { // !is_steep
            if (is_right) {
                right_theta = cv::Point2f(0, -KTheta);
                left_theta = cv::Point2f(0, KTheta);
            }
            else { // !isright
                right_theta = cv::Point2f(0, KTheta);
                left_theta = cv::Point2f(0, -KTheta);
            }
        }
    }

    bool ContoursUtils::IsWhitePixel(cv::Point2f pos, const cv::Mat& bin_img) {
        return (bin_img.at<uchar>(pos) == 0); //inverse_thresh_binaryW
    }

    bool ContoursUtils::IsImageBound(cv::Point2f pos, const cv::Mat& img) {
        int x_low = 0,
            y_low = 0,
            x_high = img.cols - 1,
            y_high = img.rows - 1;
        return (pos.x <= x_high) && (pos.y <= y_high) && (pos.x >= x_low) && (pos.y >= y_low);
    }

    ContoursUtils::ContoursUtils() {
    }

    ContoursUtils::~ContoursUtils() {
    }


}