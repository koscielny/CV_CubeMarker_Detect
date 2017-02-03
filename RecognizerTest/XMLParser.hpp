#ifndef XMLPARSER_HPP
#define XMLPARSER_HPP
#include <iostream>
#include <set>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2\opencv.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
using boost::property_tree::ptree;

    class Parser {
    public:
        ~Parser();

        static Parser *Instance(const std::string &filename) {
            static Parser instance(filename);
            //static Parser instance;
            return &instance;
        }

        cv::Mat GetIntrinsics(const std::string &in_path = "Intrin_.Logitech720p.value");
        glm::tmat3x3<double, glm::highp> GetGlmMat(const std::string &mat_path);
        std::vector<double> GetArray(const std::string &array_path);
        cv::Vec2d GetVec2(const std::string &vec2_path);
        std::vector<cv::Point2f> GetPoint2fArray(const std::string &array_path);

    private:
        ptree pt_;

        Parser(const std::string& file_path);
        Parser();
        template <typename T>
        glm::tmat3x3<T, glm::highp>
            as_mat_3x3(ptree const& pt, ptree::key_type const& path_key);
        template <typename T>
        glm::tmat4x4<T, glm::highp>
            as_mat_4x4(ptree const& pt, ptree::key_type const& path_key);
        template <typename T>
        glm::tmat2x2<T, glm::highp>
            as_mat_2x2(ptree const& pt, ptree::key_type const& path_key);
        template <typename T>
        cv::Mat_<T>
            as_mat(ptree const& pt, ptree::key_type const& path_key);
        template <typename T>
        std::vector<T>
            as_array(ptree const& pt, ptree::key_type const& path_key);
        template <typename T>
        cv::Vec<T, 2>
            as_vec2(ptree const& pt, ptree::key_type const& path_key);
        template <typename T>
        std::vector<cv::Point_<T>>
            as_point_array(ptree const& pt, ptree::key_type const& path_key);
        /*
        cv::Mat camera_intrinsics_;
        cv::Vec2d test_v2_;
        std::vector<double> test_array_;
        glm::tmat3x3<double, glm::highp> glm_mat_test;
        */
    };
#endif