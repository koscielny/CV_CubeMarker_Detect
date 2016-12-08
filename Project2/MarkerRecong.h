#pragma once
#include <opencv2\opencv.hpp>
#include <array>
#include <stack>
#include "math.h"
#include <fstream>
#include <time.h>  
#include <direct.h>
#include <io.h>

#define _DEBUG_PRINT
#define _DEBUG_FILE_DUMP
#define _DEBUG_PARAM_CHANGE
//#define _RECUR_TREE_TRAVEL
//#define MORPH_

using namespace cv;
using namespace std;

string file_time;
uint file_count = 0;
const uint8_t ASCII_ESC = 27;
const uint8_t ASCII_K = 107;
const uint8_t ASCII_G = 103	;
const uint16_t DESIRED_CAMERA_WIDTH = 640;
const uint16_t DESIRED_CAMERA_HEIGHT = 480;
const uint8_t BINARY_THRESHOLD = 230; //单阈值//越小越能滤去灰色区域//threshold = 120: 8pm日光灯，视角斜向下45度
const uint8_t BINARY_MAX = 255;
Scalar CONTOUR_COLOR = Scalar(128, 255, 255);
const uint8_t CONTOUR_MODE = CV_RETR_TREE;// CV_RETR_EXTERNAL;// CV_RETR_LIST; //CV_RETR_EXTERNAL;//outermost:CV_RETR_EXTERNAL, all without hierarchy:CV_RETR_LIST//RETR_CCOMP //RETR_TREE 
uint8_t CONTOUR_METHOD = CV_CHAIN_APPROX_SIMPLE;// CV_CHAIN_APPROX_SIMPLE;//none compress:CV_CHAIN_APPROX_NONE//
const uint8_t THRESHOLD_TYPE = THRESH_BINARY_INV; //THRESH_BINARY_INV , THRESH_BINARY
uint8_t adaptive_method = ADAPTIVE_THRESH_MEAN_C; //ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C 

int min_marker_side_length = 15; // 容忍检测到的最小边长//四边都要满足//要小于根号min_marker_size
int block_size = 1 + 1 * 5 /2 * 2;//(min_size / 4) * 2 + 1;
double offset_sub_constant = 2 * 1;//+/-/0
int min_area = 50 * 4;	
int min_marker_size = 15 * 4;
double approx_poly_eps = 0.02 * 5;
int bilateral_kernel = 4;
double sigma_color = 120;
double sigma_space = 200;  
//uint8_t last_threshold = 230;
//const double APPROX_POLY_EPS = .1;
#ifdef _DEBUG_FILE_DUMP
bool is_dump = 0;
#endif

int adaptive_method_trackingbar, block_size_trackingbar, constant_trackingbar, marker_min_size_trackingbar, approx_poly_eps_trackingbar, min_area_trackingbar;
int sigma_color_trackingbar, sigma_space_trackingbar, bilateral_kernel_trackingbar;
const int adaptive_method_count = 1, block_size_count = 100, constant_count = 30, marker_min_size_count = 30, approx_poly_eps_count = 10, min_area_count = 20;
const int sigma_C_count = 15, sigma_S_count = 15, bilateral_kernel_count = 9;
//const int sigma_color_trackingbar, sigma_space_trackingbar, bilateral_kernel_trackingbar;
class QuadNode;
class Marker;

void AddMarkers(vector<Marker>& possible_markers, const vector<vector<Point> >& contour_points_detected);
void markerDetect(vector<vector<Point>>all_contours, vector<Marker>& possible_markers, int min_side_length, int min_size);//abandened
void markerRecognize(cv::Mat& gray_frame, vector<Marker>& possible_markers, vector<Marker>& final_markers);
void DisplayMarkerVec(Mat img, vector<Marker> markers, double fps);

/*find quad tree*/
void detectCandidateContours(const vector<vector<Point> >& src_contours, vector<vector<Point> >& output_quads_contours,
	const vector<Vec4i>& src_hierarchy, vector<Vec4i> &quads_tree_hierarchy);
void RecurTreeTraversal(const vector<Vec4i>& src_hierarchy, vector<Vec4i>& quads_tree_hierarchy,
	const vector<vector<Point> >& quads_contours,
	int contour_iter, const vector <pair<bool, int> >& is_quad,
	int pre, bool parent_or_bro);/*1 parent, 0 bro */
void BroTraversal(const vector<Vec4i>& src_hierarchy, vector<Vec4i>& quads_tree_hierarchy,
	const vector<vector<Point> >& quads_contours,
	int contour_iter, const vector <pair<bool, int> >& is_quad,
	int pre, bool parent_or_bro);/*1 parent, 0 bro */
void RecurKidTraversal(const vector<Vec4i>& src_hierarchy, vector<Vec4i>& quads_tree_hierarchy,
	const vector<vector<Point> >& quads_contours,
	int contour_iter, const vector <pair<bool, int> >& is_quad,
	int pre, bool parent_or_bro);/*1 parent, 0 bro */
void StackKidTraversal(const vector<Vec4i>& src_hierarchy, vector<Vec4i>& quads_tree_hierarchy,
	const vector <pair<bool, int> >& is_quad);
bool isQuad(vector<Point> contour, vector<Point>& candidate_quad);

/*after quads found, find candidate ones*/
void detectCandidateContours2(vector<vector<Point> >& quads_contours, vector<vector<Point> >& detected_contours,
	const vector<Vec4i>& quads_tree_hierarchy, Mat& bin_img);
void FindDeeperCandidate(vector<pair<Vec4i, bool>> & tree, int tree_iter, int pre_detected,
	const vector<vector<Point> >& src_contours, vector<vector<Point> >& detected_contours,
	vector<int>& candidate_queue);
bool isCandidateContour(vector<Point>& contour, Mat& bin_img);
void bresenham_border_detect(Point end_p_a, Point end_p_b, int& good, int& dirty, Mat& bin_img);
void PixelDetect(Point& draw_pixel, Point& right_theta, Point& left_theta, int& good, int& dirty, Mat& bin_img);
void ComputeTheta(bool is_right, bool is_up, bool is_steep, Point& right_theta, Point& left_theta);
bool isWhite(Point pos, const Mat& bin_img);
bool isImageBound(Point pos, const Mat& img);

/*debug tools*/
void initUI(void );
void onChange(int, void*);
void DumpImg(const Mat &img, string);

class Marker {
public:
	Marker(int, Point2i c1, Point2i c2, Point2i c3, Point2i c4) :m_corners({ c1,c2,c3,c4 }) {};
	Marker(vector<Point> quad) :m_corners(quad) {};
	//array<Point2f, 4> m_corners;
	vector<Point2i> m_corners;
	uint8_t m_id = 0; //编码

	void DisplayCorners(Mat& img) {
		polylines(img, m_corners, 1, Scalar(255, 0, 255));
		circle(img, m_corners[1], 5, Scalar(0, 0, 255), -1);
	};
};

class QuadNode
{
public:
	QuadNode() {};
	QuadNode(Point2i v1, Point2i v2, Point2i v3, Point2i v4)
		: approx_quad({ v1, v2, v3, v4 }) {};

	vector<Point> approx_quad;
	int self;
	int child_first, parent, next = -1;//tree pointer
};


template <class SrcType, class DstType>
void ConvertVector(vector<SrcType>& src, vector<DstType>& dst) //src : Point2i, dst : Point2f
{
	cv::Mat srcMat = cv::Mat(src);
	cv::Mat dstMat = cv::Mat(dst);
	cv::Mat tmpMat;
	srcMat.convertTo(tmpMat, dstMat.type());
	dst = (vector<DstType>) tmpMat;
}
template <class SrcType, class DstType>
void ConvertVector2(vector<SrcType>& src, vector<DstType>& dst) {
	dst.resize(src.size());
	std::copy(src.begin(), src.end(), dst.begin());
}

template <class SrcType, class DstType>
void ConvertVector3(std::vector<SrcType>& src, std::vector<DstType>& dst) {
	std::copy(src.begin(), src.end(), std::back_inserter(dst));
}

#ifdef _RECUR_TREE_TRAVEL
vector<int> root_chain;
#endif

/**/
#pragma region find_quad_tree
void detectCandidateContours(const vector<vector<Point> >& src_contours, vector<vector<Point> >& output_quads_contours,
	const vector<Vec4i>& src_hierarchy, vector<Vec4i> &quads_tree_hierarchy)
{
	// hierarchy[i].first[0] , hiearchy[i][1] , hiearchy[i][2] , and hiearchy[i][3]
	// the next contour
	// the previous contour at the same hierarchical level
	// the first child contour
	// the parent contour
	// bool : is_candidate

	//获得is_quad向量 和 approx_poly_contours轮廓向量
	vector <pair<bool, int> > is_quad;//int 对应output_quads_contours中的索引
	vector<vector<Point>> quads_contours;

	is_quad.resize(src_contours.size());
	quads_contours.resize(src_contours.size());

	for (int i = 0; i < src_contours.size(); i++) {
		if (isQuad(src_contours[i], quads_contours[i])) {
			//cout << i <<endl;
			is_quad[i].first = true;
			output_quads_contours.push_back(quads_contours[i]);
			is_quad[i].second = output_quads_contours.size() - 1;
		}
		else {
			is_quad[i].first = false;
		}
	}//is_quad already , quads_contours already , output_quads_contours already
	
	quads_tree_hierarchy.resize(output_quads_contours.size(), Vec4i(-1, -1, -1, -1));
	int root_iter = 0;
	bool is_parent = 1;
	bool is_bro = 0;
	
#ifdef _RECUR_TREE_TRAVEL
	root_chain.clear();
	//没轮廓树空时，有无0号根节点？
	while (root_iter >= 0) {
			//root_iter = src_hierarchy[root_iter][0];
		RecurKidTraversal(src_hierarchy, quads_tree_hierarchy,
				quads_contours,
				root_iter, is_quad,
				-1, is_parent);//compute quads_tree_hierarchy
		root_iter = src_hierarchy[root_iter][0];//next
	}
	root_iter = 0;
	///
	BroTraversal(src_hierarchy, quads_tree_hierarchy,
		quads_contours, //output_quads_contours,
		src_hierarchy[root_iter][0], is_quad,
		root_iter, is_bro);///bro : newfound return_iter -> return_iter.nextbro///
	///
	if (0 < quads_tree_hierarchy.size()) {
		for (int i = 0; i < root_chain.size() - 1; i++) {
			quads_tree_hierarchy[is_quad[root_chain[i]].second][0] = is_quad[root_chain[i + 1]].second;
			quads_tree_hierarchy[is_quad[root_chain[i + 1]].second][1] = is_quad[root_chain[i]].second;
}

		quads_tree_hierarchy[is_quad[root_chain[root_chain.size() - 1]].second][0] = -1;
		quads_tree_hierarchy[is_quad[root_chain[0]].second][1] = -1;
	}


#else
	StackKidTraversal(src_hierarchy, quads_tree_hierarchy, is_quad);
#endif
	//for (int i = 0; i < quads_tree_hierarchy.size(); i++)
	//	cout << quads_tree_hierarchy[i] << endl;
}

#ifdef _RECUR_TREE_TRAVEL
#pragma region USED_RECUR_METHOD
//return contours&tree_hierarchy of candidate quads
//using bool isQuad(contour, &candidate_quad);
//if isQuad, pushback contour and reorganize the hierarchy
//DFS traversal

void RecurTreeTraversal(const vector<Vec4i>& src_hierarchy, vector<Vec4i>& quads_tree_hierarchy,
	const vector<vector<Point> >& quads_contours, //vector<vector<Point> >& output_quads_contours,
	int contour_iter, const vector <pair<bool, int> >& is_quad,
	int pre, bool parent_or_bro)/*1 parent, 0 bro */
{

	if (parent_or_bro == 0){//bro
		int next_iter = contour_iter;
		int return_iter = -1;
		while (next_iter >= 0 && return_iter == -1){
			if (is_quad[next_iter].first == true) {
				return_iter = next_iter;
				next_iter = src_hierarchy[next_iter][0];
				RecurTreeTraversal(src_hierarchy, quads_tree_hierarchy,
					quads_contours, //output_quads_contours,
					next_iter, is_quad,
					return_iter, 0);///bro : newfound return_iter -> return_iter.nextbro///
				break;//return_iter = next_iter do the same jumping out function
			}
			next_iter = src_hierarchy[next_iter][0];//next
		}
		if (return_iter != -1) {
			if(pre != -1){
				cout << "bro";
				quads_tree_hierarchy[is_quad[pre].second][0] = is_quad[return_iter].second;
				quads_tree_hierarchy[is_quad[return_iter].second][1] = is_quad[pre].second;
			}
			else {//root node 
				quads_tree_hierarchy[is_quad[return_iter].second][1] = -1;
				cout << "error tree 1" << endl;//root node ought not to find bro
			}
		}
		else {
			if (pre != -1){
				quads_tree_hierarchy[is_quad[pre].second][0] = -1;
			}
			else {
				cout << "error tree 2" << endl;
			}
		}
		//hierarchy[pre].bro = return_iter;hierarchy[return_iter].bigbro = pre
	}
	else {//if(parent_or_bro == 1)//parent = pre, return contour_iter
		if (contour_iter < 0)
			cout << "tree error 3" << endl;//没轮廓树空时，有无0号根节点？
		else {
			if (is_quad[contour_iter].first == true){
				//找第一个子，以及告知其所有子设置父节点（pre，第一子flag）
				//int quad_iter = is_quad[contour_iter].second;//在新队列中的id
				if (pre == -1) {
					root_chain.push_back(contour_iter);
					quads_tree_hierarchy[is_quad[contour_iter].second][3] = -1;
				}
				else {
					cout << "parent";
					quads_tree_hierarchy[is_quad[contour_iter].second][3] = is_quad[pre].second;//parent
					if (quads_tree_hierarchy[is_quad[pre].second][2] == -1)//no former child
						quads_tree_hierarchy[is_quad[pre].second][2] = is_quad[contour_iter].second;//firstchild
				}

				int next_iter = src_hierarchy[contour_iter][2];//first child
				if (next_iter >= 0) {//not deepest//have child
					quads_tree_hierarchy[is_quad[next_iter].second][1] = -1;
					RecurTreeTraversal(src_hierarchy, quads_tree_hierarchy,
						quads_contours, //output_quads_contours,
						src_hierarchy[next_iter][0], is_quad,
						next_iter, 0);///bro : contour_iter.firstchild -> fisrtchild.nextbro
					while (next_iter >= 0) {//if exist , >0 //get next bro
						int temp = next_iter;						
						next_iter = src_hierarchy[temp][0];//next bro
						RecurTreeTraversal(src_hierarchy, quads_tree_hierarchy,
							quads_contours, //output_quads_contours,
							temp, is_quad ,
							contour_iter, 1);///parent : contour_iter -> contour.recur all child
					}
					//no more bro
				}
				else {//no child, already cover pre == -1 situation
						quads_tree_hierarchy[is_quad[contour_iter].second][2] = -1;//no child
					//else cout << "new branch empty" << endl;
				}
			}
			else {//需要跳过的//但要借助遍历
				int next_iter = src_hierarchy[contour_iter][2];//first child
				if (next_iter < 0) {//no child
					if (pre != -1){
						//quads_tree_hierarchy[is_quad[pre].second][2] = -1;//no child
					}
					else {
						//cout << "new branch empty" << endl;//no pre no child branch
					}
				}
				else {//not deepest//have child
					while (next_iter >= 0) {//if exist , >0 //get next bro
						int temp = next_iter;
						next_iter = src_hierarchy[temp][0];//next bro
						RecurTreeTraversal(src_hierarchy, quads_tree_hierarchy,
							quads_contours, //output_quads_contours,
							temp, is_quad,
							pre, 1);///parent : pre -> pre.recur all child
					}
					//no more bro
				}
			}
		}
	}
}


void BroTraversal(const vector<Vec4i>& src_hierarchy, vector<Vec4i>& quads_tree_hierarchy,
	const vector<vector<Point> >& quads_contours, //vector<vector<Point> >& output_quads_contours,
	int contour_iter, const vector <pair<bool, int> >& is_quad,
	int pre, bool parent_or_bro)/*1 parent, 0 bro */
{//bro
	vector<int> bro_chain;
	int next_iter = contour_iter;

	while (next_iter >= 0) {
		if (is_quad[next_iter].first == true) {
			bro_chain.push_back(next_iter);
		}
		next_iter = src_hierarchy[next_iter][0];
	}

	if (bro_chain.size() > 0) {
		for (int i = 0; i < bro_chain.size() - 1; i++) {
			quads_tree_hierarchy[is_quad[bro_chain[i]].second][0] = is_quad[bro_chain[i + 1]].second;
			quads_tree_hierarchy[is_quad[bro_chain[i + 1]].second][1] = is_quad[bro_chain[i]].second;
		}

		quads_tree_hierarchy[is_quad[bro_chain[bro_chain.size() - 1]].second][0] = -1;
		quads_tree_hierarchy[is_quad[bro_chain[0]].second][1] = -1;
	}
}

void RecurKidTraversal(const vector<Vec4i>& src_hierarchy, vector<Vec4i>& quads_tree_hierarchy,
	const vector<vector<Point> >& quads_contours, //vector<vector<Point> >& output_quads_contours,
	int contour_iter, const vector <pair<bool, int> >& is_quad,
	int pre, bool parent_or_bro)/*1 parent, 0 bro */
{
	if (contour_iter >= 0) {
		if (is_quad[contour_iter].first == true) {
			if (pre == -1) {
				root_chain.push_back(contour_iter);//need a global root_chain and clear it everytime
			}
			else {
				cout << "parent";
				quads_tree_hierarchy[is_quad[contour_iter].second][3] = is_quad[pre].second;//parent
				if (quads_tree_hierarchy[is_quad[pre].second][2] == -1)//no former child
					quads_tree_hierarchy[is_quad[pre].second][2] = is_quad[contour_iter].second;//firstchild
			}

			int next_iter = src_hierarchy[contour_iter][2];//first child
			if (next_iter >= 0) {//not deepest//have child
				BroTraversal(src_hierarchy, quads_tree_hierarchy,
					quads_contours, //output_quads_contours,
					src_hierarchy[next_iter][0], is_quad,
					next_iter, 0);///bro : contour_iter.firstchild -> fisrtchild.nextbro
				while (next_iter >= 0) {//if exist , >0 //get next bro
					int temp = next_iter;
					next_iter = src_hierarchy[temp][0];//next bro
					RecurKidTraversal(src_hierarchy, quads_tree_hierarchy,
						quads_contours, //output_quads_contours,
						temp, is_quad,
						contour_iter, 1);///parent : contour_iter -> contour.recur all child
				}
			}
		}
		else {//需要跳过的//但要借助遍历
			int next_iter = src_hierarchy[contour_iter][2];//first child
			//if (next_iter >= 0) {//not deepest//have child
			while (next_iter >= 0) {//if exist , >0 //get next bro
				int temp = next_iter;
				next_iter = src_hierarchy[temp][0];//next bro
				RecurKidTraversal(src_hierarchy, quads_tree_hierarchy,
					quads_contours, //output_quads_contours,
					temp, is_quad,
					pre, 1);///parent : pre -> pre.recur all child
			}
		}
	}	
	else {//contour_iter<0
		cout << "tree error 3" << endl;//findcontour没轮廓树空时
	}
}
#pragma endregion
#endif

	// 0 next 
	// 1 previous 
	// 2 first 
	// 3 parent 
void StackKidTraversal(const vector<Vec4i>& src_hierarchy, vector<Vec4i>& quads_tree_hierarchy, 
	const vector <pair<bool, int> >& is_quad)/*1 parent, 0 bro */
{
	vector<int> root_chain;

	/* pair(current, parent)*/
	stack<pair<int, int>> src_tree_stack;
	int next_bro = 0;
	int root_parent = -1;

	while (next_bro >= 0){
		src_tree_stack.push({next_bro, root_parent });
		next_bro = src_hierarchy[next_bro][0];
	}

	while (!src_tree_stack.empty()) {//main loop
		pair<int, int> cur_iter = src_tree_stack.top();
		int cur_node = cur_iter.first;
		int cur_parent = cur_iter.second;
		int parent_of_kids;
		src_tree_stack.pop();

		if (is_quad[cur_node].first == true) {/*is quad node*/
			if (cur_parent == -1) {
				root_chain.push_back(cur_node);
			}
			else {
				cout << "parent";
				quads_tree_hierarchy[is_quad[cur_node].second][3] = is_quad[cur_parent].second;//parent
				int last_kid = quads_tree_hierarchy[is_quad[cur_parent].second][2];//first
				if (last_kid == -1) {//no former child
					quads_tree_hierarchy[is_quad[cur_parent].second][2] = is_quad[cur_node].second;//firstchild
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
			src_tree_stack.push({next_kid, parent_of_kids});
			next_kid = src_hierarchy[next_kid][0];//kid's bro
		}
	}//end stack while

	if (0 < quads_tree_hierarchy.size()) {
		for (int i = 0; i < root_chain.size() - 1; i++) {
			quads_tree_hierarchy[is_quad[root_chain[i]].second][1] = is_quad[root_chain[i + 1]].second;
			quads_tree_hierarchy[is_quad[root_chain[i + 1]].second][0] = is_quad[root_chain[i]].second;
		}

		quads_tree_hierarchy[is_quad[root_chain[root_chain.size() - 1]].second][1] = -1;
		quads_tree_hierarchy[is_quad[root_chain[0]].second][0] = -1;
	}
}

bool isQuad(vector<Point> contour, vector<Point>& candidate_quad)
{
	vector<Point> approx_poly;
	double eps = contour.size()*approx_poly_eps;//eps的设置
	approxPolyDP(contour, approx_poly, eps, true);//
	//凸四边形
	if (contourArea(contour) > min_area) {
		if (approx_poly.size() == 4 && isContourConvex(approx_poly)) {
			//Sort the points in anti-clockwise  
			candidate_quad = { approx_poly[0], approx_poly[1], approx_poly[2], approx_poly[3] };
			Point2i v1 = candidate_quad[1] - candidate_quad[0];
			Point2i v2 = candidate_quad[2] - candidate_quad[0];
			if (v1.cross(v2) > 0)    //由于图像坐标的Y轴向下，所以大于零才代表逆时针  
			{
				swap(candidate_quad[1], candidate_quad[3]);
			}
#ifdef _DEBUG_FILE_DUMP
		if (is_dump) {
			ofstream dump_file;
			//string time = "";
			string log_path = "D:\\Dump_Folder\\dump_" + file_time + "\\__dump.txt";
			cout << file_time;
			dump_file.open(log_path, ios::out | ios::app);
			//dump_file << possible_markers.size() << endl;
			dump_file << "Position(" << candidate_quad[0].x << ", " << candidate_quad[0].y << ")" << endl;
			dump_file << "Side_ab(" << sqrt((candidate_quad[1] - candidate_quad[0]).dot(candidate_quad[1] - candidate_quad[0])) << endl;
			dump_file << "Side_bc(" << sqrt((candidate_quad[2] - candidate_quad[1]).dot(candidate_quad[2] - candidate_quad[1])) << endl;
			dump_file.close();
		}
#endif // _DEBUG_FILE_DUMP
				return true;
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}
}
#pragma endregion

#pragma region reduce_quads_into_candidate


void detectCandidateContours2(vector<vector<Point> >& quads_contours, vector<vector<Point> >& detected_contours,
	const vector<Vec4i>& quads_tree_hierarchy, Mat& bin_img)
{
	if (quads_contours.size() > 0) {
	vector<pair<Vec4i, bool>> CandidateTree;

	for (int i = 0; i < quads_contours.size(); i++) {
		CandidateTree.push_back(make_pair(quads_tree_hierarchy[i], false));
		if (isCandidateContour(quads_contours[i], bin_img)) {
			CandidateTree[i].second = true;
			cout << quads_tree_hierarchy[i] << endl;
		}
		//hierarchy failed
	}
	vector<int> candidate_queue;
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

void FindDeeperCandidate(vector<pair<Vec4i, bool>> & tree, int tree_iter, int pre_detected,
	const vector<vector<Point> >& src_contours, vector<vector<Point> >& detected_contours,
	vector<int>& candidate_queue)
{
	int cur_detected;

	if (tree[tree_iter].second == true)
		cur_detected = tree_iter;//更深的会替代上层的
	else
		cur_detected = pre_detected;

	int next_iter = tree[tree_iter].first[2];//first child
	if (next_iter >= 0) {//not deepest
		while (next_iter >= 0) {//if exist , >0
			FindDeeperCandidate(tree, next_iter, cur_detected, src_contours, detected_contours, candidate_queue);
			next_iter = tree[next_iter].first[0];//next bro
		}
	}
	else {//already deepest
		  //conclude candidate
		if (cur_detected >= 0) {
			if (find(candidate_queue.begin(), candidate_queue.end(), cur_detected) == candidate_queue.end())//相同重复
			{
				candidate_queue.push_back(cur_detected);
				detected_contours.push_back(src_contours[cur_detected]);
			}
		}
	}
}

bool isCandidateContour(vector<Point>& contour, Mat& bin_img) //const or not
{//Bresenham:for(every_side_in_poly)
	int good = 0, dirty = 0;

	//convert into counter clockwise 
	vector<Point> clock_wise_contour;
	convexHull(contour, clock_wise_contour, true);//const or not // points sequence order changed
	for (int i = 0; i < contour.size(); i++)
	{
		bresenham_border_detect( contour[i], contour[(i+1) % 4], good, dirty, bin_img);//统计下good/dirty比例//i-1 到 i 逆时针，
	}

	if (((double)(good + 1) / (double)(good + dirty + 1)) > 0.6 && ( (good + dirty)>0)) {
		//cout << "__g" << good << " d" << dirty << endl;
		return true;
	}
	else {
		return false;
	}
}

void bresenham_border_detect(Point start_p, Point end_p, int& good, int& dirty, Mat& bin_img)
{
	//a -> b : clockwise , right side of a->b vector : white,  left side of a->b vector : black
	int x0 = start_p.x,
		y0 = start_p.y,
		x1 = end_p.x,
		y1 = end_p.y;
	Point draw_pixel = start_p;
	int dx,             // difference in x's
		dy,             // difference in y's
		dx2,            // dx,dy * 2
		dy2,
		x_inc,          // amount in pixel space to move during drawing
		y_inc,          // amount in pixel space to move during drawing
		error,          // the discriminant i.e. error i.e. decision variable
		index;          // only used for looping

	Point right_theta;		//vector directing to right side, to compute right-side / left-side point
	Point left_theta;
	dx = x1 - x0;
	dy = y1 - y0;
	// test which direction the line is going in i.e. slope angle
	if (dx >= 0) {
		x_inc = 1;
	} // end if line is moving right
	else {
		x_inc = -1;
		dx = -dx;  // absolute value
	} // end else moving left

	if (dy >= 0) {
		y_inc = 1;
	} // end if moving up
	else {
		y_inc = -1;
		dy = -dy;  // absolute value
	} // end else moving down

	dx2 = dx << 1;
	dy2 = dy << 1;

	ComputeTheta((x_inc > 0), (y_inc > 0), (dx <= dy), right_theta, left_theta);

	// now based on which delta is greater we can draw the line
	//|slope| <= 1 //theta follow y-direction
	if (dx > dy) {
		error = dy2 - dx;
		// draw the line
		for (index = 0; index <= dx; index++) {
			// test if error has overflowed
			if (error >= 0) {
				error -= dx2;
				// move to next line
				draw_pixel.y += y_inc;
			} // end if error overflowed
			  // adjust the error term
			error += dy2;
			// move to the next pixel
			draw_pixel.x += x_inc;
			PixelDetect(draw_pixel, right_theta, left_theta, good, dirty, bin_img);
		} // end for
	}
	// |slope| > 1 //theta follow x-direction
	else {
		error = dx2 - dy;
		// draw the line
		for (index = 0; index <= dy; index++) {
			// test if error overflowed
			if (error >= 0) {
				error -= dy2;
				// move to next line
				draw_pixel.x += x_inc;
			} // end if error overflowed
			error += dx2;
			// move to the next pixel
			draw_pixel.y += y_inc;
			PixelDetect(draw_pixel, right_theta, left_theta, good, dirty, bin_img);
		} // end for
	}
	
}

void PixelDetect(Point& draw_pixel, Point& right_theta, Point& left_theta, int& good, int& dirty ,Mat& bin_img){
	if (isImageBound(draw_pixel + right_theta, bin_img) && isImageBound(draw_pixel + left_theta, bin_img)) {
		if (isWhite(draw_pixel + right_theta, bin_img) && !isWhite(draw_pixel + left_theta, bin_img)) {
			good++;
		}
		else {
			dirty++;
		}
	}
}

#define theta 1
void ComputeTheta(bool is_right, bool is_up, bool is_steep, Point& right_theta, Point& left_theta) {
	if (is_up && is_steep) {//isright or isleft
		right_theta = Point(theta, 0);
		left_theta = Point(-theta, 0);
	}
	if (is_right && !is_steep) {
		right_theta = Point(0, -theta);
		left_theta = Point(0, theta);
	}
	if (!is_up && is_steep) {
		right_theta = Point(-theta, 0);
		left_theta = Point(theta, 0);
	}
	if (!is_right && !is_steep) {
		right_theta = Point(0, theta);
		left_theta = Point(0, -theta);
	}
}

bool isWhite(Point pos, const Mat& bin_img) {
	return (bin_img.at<uchar>(pos) == 0); //inverse_thresh_binary

}

#pragma endregion

void AddMarkers(vector<Marker>& possible_markers, const vector<vector<Point> >& contour_points_detected) {
	for (int i = 0; i < contour_points_detected.size(); i++) {
		possible_markers.push_back(Marker(contour_points_detected[i]));
	}
}


bool isImageBound(Point pos, const Mat& img) {
	int x_low = 0,
		y_low = 0,
		x_high = img.cols - 1,
		y_high = img.rows - 1;
	return (pos.x <= x_high) && (pos.y <= y_high) && (pos.x >= x_low) && (pos.y >= y_low);
}


void initUI(void)
{
	namedWindow("GrayCamera");
	namedWindow("BinaryCamera");
	namedWindow("ContourCamera");
	namedWindow("MarkerRocog");
	namedWindow("Toolbox", WINDOW_NORMAL);

	createTrackbar("策略自适应阈值_adaptive_method_trackingbar", "Toolbox", //推荐值0
		&adaptive_method_trackingbar, adaptive_method_count, onChange, 0);
	createTrackbar("块大小阈值化_block_size_trackingbar", "Toolbox", //推荐值 6 或 15 或 24以上
		&block_size_trackingbar, block_size_count, onChange, 0);
	createTrackbar("2*偏移阈值_constant_trackingbar", "Toolbox", //推荐值 > 2 * 1 , 3 或 8 或12
		&constant_trackingbar, constant_count, onChange, 0);
	//createTrackbar("8*Mark大小_marker_min_size_trackingbar", "BinaryCamera",//推荐值 尽量大于15*8//另一种说法约等于8 * block_size_trackbar = 2 * block_size 
	//	&marker_min_size_trackingbar, marker_min_size_count, onChange, 0);
	createTrackbar("0.1*EPS_approx_poly_eps_trackingbar", "Toolbox",//推荐值 >0.01
		&approx_poly_eps_trackingbar, approx_poly_eps_count, onChange, 0);
	createTrackbar("50*AREA_min_area_trackingbar", "Toolbox",//推荐值 >0.01
		&min_area_trackingbar, min_area_count, onChange, 0);
	createTrackbar("颜色系数_sigma_color_trackingbar",
		"Toolbox", &sigma_color_trackingbar, sigma_C_count, onChange, 0
	);
	createTrackbar("距离系数_sigma_space_trackingbar",
		"Toolbox", &sigma_space_trackingbar, sigma_S_count, onChange, 0
	);
	createTrackbar("双边核大小_bilateral_kernel_trackingbar",
		"Toolbox", &bilateral_kernel_trackingbar, bilateral_kernel_count, onChange, 0
	);
}

void onChange(int boo, void*)
{
	//adaptive_method_trackingbar
	switch (adaptive_method_trackingbar) {
	case (0):
		adaptive_method = ADAPTIVE_THRESH_GAUSSIAN_C;
		break;
	case (1):
		adaptive_method = ADAPTIVE_THRESH_MEAN_C;
		break;
	default:
		break;
	}
	//marker_min_size
	min_marker_size = marker_min_size_trackingbar * 8;
	//block_size_trackingbar
	block_size = block_size_trackingbar / 2 * 2 + 1;
	if (block_size < 3)		block_size = 3;
	//constant_trackingbar
	offset_sub_constant = constant_trackingbar * 2;
	//
	approx_poly_eps = approx_poly_eps_trackingbar * 0.02;
	min_area = min_area_trackingbar * 50;
	//bilateral
	sigma_color = 30.0 * sigma_color_trackingbar;
	sigma_space = 30.0 * sigma_space_trackingbar;
	bilateral_kernel = bilateral_kernel_trackingbar >= 3 ? bilateral_kernel_trackingbar : 3;
}

void DumpImg(const Mat &img, string img_name)
{
	if (is_dump) {
		//file_count++;
		string dirstr = "D:\\Dump_Folder\\dump_" + file_time;
		const char *dir = dirstr.c_str();
		if (_access(dir, 0) != 0) {
			_mkdir(dir);
			cout << "success!" << endl;
		}
#ifdef MORPH_
		string img_path = "D:\\Dump_Folder\\dump_" + file_time + "\\__" + to_string(file_count + 1) + img_name + ".bmp";
#else
		string img_path = "D:\\Dump_Folder\\dump_" + file_time + "\\__" + to_string(file_count + 1) + img_name + "NoMorph" + ".bmp";
#endif
		cv::imwrite(img_path, img);
	}
}