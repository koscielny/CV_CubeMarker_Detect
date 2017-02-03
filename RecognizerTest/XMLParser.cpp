#include "prefix.hpp"
#include "XMLParser.hpp"

//using boost::property_tree::ptree;
    Parser::Parser() {
        boost::property_tree::read_xml("config.xml", pt_);

    }
    Parser::Parser(const std::string& file_path) {
        boost::property_tree::read_xml(file_path, pt_);
    }

    Parser::~Parser() {
    }

    cv::Mat Parser::GetIntrinsics(const std::string &in_path) {
        return as_mat<double>(pt_, in_path);
    }
    glm::tmat3x3<double, glm::highp> Parser::GetGlmMat(const std::string &mat_path) {
        return as_mat_3x3<double>(pt_, mat_path);
    }
    std::vector<double> Parser::GetArray(const std::string &array_path) {
        return as_array<double>(pt_, array_path);
    }
    std::vector<cv::Point2f> Parser::GetPoint2fArray(const std::string &array_path) {
        return as_point_array<float>(pt_, array_path);
    }

    cv::Vec2d Parser::GetVec2(const std::string &vec2_path) {
        return as_vec2<double>(pt_, vec2_path);
    }
    // 最多再写个std::set<std::pair<cv::Mat *, std::string>> m_param_matrix; 的reader
    // T = double
    // string = ptree::key_type const& key
    template <typename T>
    glm::tmat3x3<T, glm::highp>
        Parser::as_mat_3x3(ptree const& pt, ptree::key_type const& path_key) {
        glm::tmat3x3<T, glm::highp> r;
        std::string str = pt.get<std::string>(path_key);
        std::vector<std::string> vec_str;

        boost::algorithm::split(vec_str, str, boost::algorithm::is_any_of(","));
        for (size_t i = 0; i < 3; i++) {
            std::istringstream cvt_ss(vec_str[i]);
            // boost::algorithm::split(row_str, vec_str[i], boost::algorithm::is_any_of(" "));
            for (size_t j = 0; j < 3; j++) {
                cvt_ss >> r[i][j];
            }
        }
        return r;
    }

    template <typename T>
    glm::tmat4x4<T, glm::highp>
        Parser::as_mat_4x4(ptree const& pt, ptree::key_type const& path_key) {
        glm::tmat4x4<T, glm::highp> r;
        std::string str = pt.get<std::string>(path_key);
        std::vector<std::string> vec_str;

        boost::algorithm::split(vec_str, str, boost::algorithm::is_any_of(","));
        for (size_t i = 0; i < 4; i++) {
            std::istringstream cvt_ss(vec_str[i]);
            for (size_t j = 0; j < 4; j++) {
                cvt_ss >> r[i][j];
            }
        }
        return r;
    }

    template <typename T>
    glm::tmat2x2<T, glm::highp>
        Parser::as_mat_2x2(ptree const& pt, ptree::key_type const& path_key) {
        glm::tmat2x2<T, glm::highp> r;
        std::string str = pt.get<std::string>(path_key);
        std::vector<std::string> vec_str;

        boost::algorithm::split(vec_str, str, boost::algorithm::is_any_of(","));
        for (size_t i = 0; i < 2; i++) {
            std::istringstream cvt_ss(vec_str[i]);
            for (size_t j = 0; j < 2; j++) {
                cvt_ss >> r[i][j];
            }
        }
        return r;
    }

    // 不定长Matrix, 利用 *data , cols, rows 初始化
    template <typename T>
    cv::Mat_<T>
        Parser::as_mat(ptree const& pt, ptree::key_type const& path_key) {
        T* data;
        T v;
        std::string str = pt.get<std::string>(path_key);
        std::vector<std::string> vec_str;
        boost::algorithm::split(vec_str, str, boost::algorithm::is_any_of(","));
        int rows = vec_str.size();
        // !boost::algorithm::split(row0_str, vec_str[0], ',');
        // !cols = row0_str.size();
        int cols = 0;
        if (rows > 0) {
            std::istringstream cvt_ss(vec_str[0]);
            while (cvt_ss >> v)
                cols++;
        }
        data = new T[rows * cols];
        // !data = (T *)malloc(rows * cols * sizeof(T));

        for (size_t i = 0; i < rows; i++) {
            std::istringstream cvt_ss(vec_str[i]);
            for (size_t j = 0; j < cols; j++) {
                cvt_ss >> data[rows * i + j];
            }
        }
        cv::Mat_<T> r(rows, cols, data);
        return r;
    }

    template <typename T>
    std::vector<T>
        Parser::as_array(ptree const& pt, ptree::key_type const& path_key) {
        std::vector<T> r;
        std::string str = pt.get<std::string>(path_key);
        T v;
        std::istringstream cvt_ss(str);
        while (cvt_ss >> v)
            r.push_back(v);
        return r;
    }

    template <typename T>
    std::vector<cv::Point_<T>>
        Parser::as_point_array(ptree const& pt, ptree::key_type const& path_key) {
        std::vector<cv::Point_<T>> r;
        std::string str = pt.get<std::string>(path_key);
        cv::Point_<T> p;
        std::vector<std::string> vec_str;
        boost::algorithm::split(vec_str, str, boost::algorithm::is_any_of(";"));
        int rows = vec_str.size();
        for (auto str_seg : vec_str) {
            std::istringstream cvt_ss(str_seg);
            cvt_ss >> p.x >> p.y;
            r.push_back(p);
        }
        return r;
    }

    template <typename T>
    cv::Vec<T, 2> Parser::as_vec2(ptree const& pt, ptree::key_type const& path_key) {
        std::string str = pt.get<std::string>(path_key);
        T v1, v2;
        std::istringstream cvt_ss(str);
        cvt_ss >> v1 >> v2;
        return cv::Vec<T, 2>(v1, v2);
    }
