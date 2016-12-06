#include "MarkerRecong.h"


int main(int argc, char* argv[])
{
	time_t time_now = time(0);
	tm *p_time_now = localtime(&time_now);
	char time_buff[120] = { 0 };
	strftime(time_buff, 120, "%d_%m_%Y_%H_%M_%S", p_time_now);
	file_time = string(time_buff);

	double fps_tick_count = 0;
	double fps;
	//从摄像头读入视频  
	VideoCapture videoCapture(0);
	if (!videoCapture.isOpened())
	{
		return -1;
	}

	//循环显示每一帧  
	videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);
	bool stop = false;
	vector<Marker> possible_markers, final_markers;
	Mat bgr_frame, gray_frame, binary_frame, binary_frame_2;// canny_frame, filtered_binary_frame, threshold_filtered_binary_frame, eq_filtered_binary_frame;//用于存储摄像头读取的每一帧图像  
	vector<vector<Point> > contour_points_0;
	contour_points_0.clear();
	//vector<vector<Point> > contour_points_filtered_by_area;
	vector<vector<Point> > contour_points_detected;
	contour_points_detected.clear();
	initUI();

	while (!stop)
	{
		fps_tick_count = (double)cv::getTickCount();
		//BGR
		videoCapture >> bgr_frame;
		//Gray
		cvtColor(bgr_frame, gray_frame, CV_BGR2GRAY);
		imshow("GrayCamera", gray_frame);
		//Binary
		//markerDetect(Mat& gray_frame, vector<Marker>& possible_markers, block_size, offset_sub_constant, min_side_length);
		adaptiveThreshold(gray_frame, binary_frame, BINARY_MAX, adaptive_method, THRESHOLD_TYPE, block_size, offset_sub_constant);
		bilateralFilter(binary_frame,
			binary_frame_2,
			bilateral_kernel,
			sigma_color,
			sigma_space
		);
		imshow("BinaryCamera12", binary_frame_2);
#ifdef MORPH_
		morphologyEx(binary_frame, binary_frame, MORPH_OPEN, Mat());
#endif
		imshow("BinaryCamera", binary_frame); 
#ifdef _DEBUG_FILE_DUMP
		DumpImg(binary_frame, "thresh");
		DumpImg(bgr_frame, "color");
#endif
		//Canny(filtered_binary_frame, canny_frame, 0, 0);
		//Contours
		contour_points_0.clear();
		contour_points_detected.clear();
		//contour_points_filtered_by_area.clear();
		vector<Vec4i> contour_hierarchy;
		vector<Vec4i> quads_tree_hierarchy;
		vector<vector<Point> > output_quads_contours;
		// hierarchy[i][0] , hiearchy[i][1] , hiearchy[i][2] , and hiearchy[i][3]
		// the next contour
		// the previous contour at the same hierarchical level
		// the first child contour
		// the parent contour

		findContours(binary_frame, contour_points_0, contour_hierarchy, RETR_TREE, CONTOUR_METHOD);
		//findContours(binary_frame, contour_points_0, CONTOUR_MODE, CONTOUR_METHOD);//input:binary
		//detect candidate before filter : increase the computational complexity
		//While it's hard to maintain hierarchy as filtering 
		if (contour_points_0.size() < 100000)
			detectCandidateContours(contour_points_0, output_quads_contours, contour_hierarchy, quads_tree_hierarchy);
		cout << output_quads_contours.size() << "\t候选四边形 A" << endl;
		Mat quad_frame(bgr_frame.size(), CV_8UC3, Scalar(0));

		drawContours(quad_frame, output_quads_contours, -1, CONTOUR_COLOR);

		imshow("QuadCamera", quad_frame);
		adaptiveThreshold(gray_frame, binary_frame, BINARY_MAX, adaptive_method, THRESHOLD_TYPE, block_size, offset_sub_constant);
		detectCandidateContours2(output_quads_contours, contour_points_detected, quads_tree_hierarchy, binary_frame);

		//TO DELETE
		//for (int i = 0; i < contour_points_0.size(); i++) {
		//	if (contourArea(contour_points_0[i]) > min_area)
		//		contour_points_filtered_by_area.push_back(contour_points_0[i]);
		//}
		int contour_index = -1;
		Mat contour_frame(bgr_frame.size(), CV_8UC3, Scalar(0));
		//drawContours(contour_frame, contour_points_filtered_by_area, contour_index, CONTOUR_COLOR, 1, LINE_8, contour_hierarchy);//hierarchy

		drawContours(contour_frame, contour_points_detected, contour_index, CONTOUR_COLOR);
		imshow("ContourCamera", contour_frame);
#ifdef _DEBUG_FILE_DUMP
		DumpImg(contour_frame, "contour");
#endif
		possible_markers.clear();
		//markerDetect(contour_points_filtered_by_area, possible_markers, min_marker_side_length, min_marker_size);
		AddMarkers(possible_markers, contour_points_detected);
#ifdef _DEBUG_PRINT
		cout << possible_markers.size() <<"\t候选四边形" << endl;
#endif // !_DEBUG_PRINT
		fps_tick_count = ((double)cv::getTickCount() - fps_tick_count) / getTickFrequency();
		fps = 1.0 / fps_tick_count;
		markerRecognize(gray_frame,  possible_markers, final_markers);
		DisplayMarkerVec(bgr_frame, possible_markers, fps);

		char key = waitKey(30);
		if (key == ASCII_ESC)
			break;

#ifdef _DEBUG_FILE_DUMP
		is_dump = 0;
		if (key == ASCII_K)
			is_dump = 1;
#endif
#ifdef _DEBUG_PARAM_CHANGE
		//CONTOUR_METHOD = CV_CHAIN_APPROX_SIMPLE;
		if (key == ASCII_G){
			CONTOUR_METHOD = 3 - CONTOUR_METHOD;//CV_CHAIN_APPROX_NONE;
			if(CONTOUR_METHOD == CV_CHAIN_APPROX_SIMPLE)
				cout << "simple"<<endl;
		}
#endif
	}
	return 0;
}

void AddMarkers(vector<Marker>& possible_markers, const vector<vector<Point> >& contour_points_detected) {
	for (int i = 0; i < contour_points_detected.size(); i++) {
		possible_markers.push_back(Marker(contour_points_detected[i]));

	}
}

#pragma region USED_DETECT_METHOD
void markerDetect(vector<vector<Point>>contours, vector<Marker>& possible_markers, int min_side_length, int min_size)
{
	vector<Point> approx_poly;
	for (int i = 0; i < contours.size(); ++i)
	{
		double eps = contours[i].size()*approx_poly_eps;//eps的设置
		approxPolyDP(contours[i], approx_poly, eps, true);//

		//四边形
		if (approx_poly.size() != 4)
			continue;

		//凸
		if (!isContourConvex(approx_poly))
			continue;

		//Ensure that the distance between consecutive points is large enough  //边长较大 min_side_length
		int min_side = INT_MAX;
		for (int j = 0; j < 4; ++j)
		{
			Point2i side = approx_poly[j] - approx_poly[(j + 1) % 4];
			min_side = min(min_side, side.dot(side));
		}
		if (min_side < min_side_length * min_side_length)
			continue;

		//Sort the points in anti-clockwise  
		Marker marker = Marker(0, approx_poly[0], approx_poly[1], approx_poly[2], approx_poly[3]);
		Point2i v1 = marker.m_corners[1] - marker.m_corners[0];
		Point2i v2 = marker.m_corners[2] - marker.m_corners[0];
		if (v1.cross(v2) > 0)    //由于图像坐标的Y轴向下，所以大于零才代表逆时针  
		{
			swap(marker.m_corners[1], marker.m_corners[3]);
		}
		possible_markers.push_back(marker);
#ifdef _DEBUG_FILE_DUMP
		if(is_dump){
			ofstream dump_file;
			//string time = "";
			string log_path = "D:\\Dump_Folder\\dump_" + file_time + "\\__dump.txt" ;
			cout << file_time;
			dump_file.open(log_path, ios::out | ios::app);
			dump_file << possible_markers.size() << endl;
			dump_file << "Position(" << marker.m_corners[0].x << ", " << marker.m_corners[0].y << ")" << endl;
			dump_file << "Side_ab(" << sqrt((marker.m_corners[1] - marker.m_corners[0]).dot(marker.m_corners[1] - marker.m_corners[0])) << endl;
			dump_file << "Side_bc(" << sqrt((marker.m_corners[2] - marker.m_corners[1]).dot(marker.m_corners[2] - marker.m_corners[1])) << endl;
			dump_file.close();
		}
#endif // _DEBUG_FILE_DUMP

	}
}
#pragma endregion
const int MARKER_SIZE = 200, MARKER_CELL_SIZE = MARKER_SIZE / 8;
vector<Point2f> m_marker_coords = { Point2f(0,0),Point2f(MARKER_SIZE - 1, 0), Point2f(MARKER_SIZE - 1, MARKER_SIZE - 1),  Point2f(0, MARKER_SIZE - 1) };

void markerRecognize(cv::Mat& gray_frame, vector<Marker>& possible_markers, vector<Marker>& final_markers)
{
	final_markers.clear();

	Mat marker_image = Mat::zeros(MARKER_SIZE, MARKER_SIZE, CV_8UC1);
	Mat bit_matrix(5, 5, CV_8UC1);

	for (int i = 0; i < possible_markers.size(); ++i)
	{
		vector<Point2f> m_corners_f;	
		Mat(possible_markers[i].m_corners).convertTo(m_corners_f, Mat(m_corners_f).type());
		//ConvertVector2(possible_markers[i].m_corners, m_corners_f);

		Mat M = getPerspectiveTransform(m_corners_f, m_marker_coords);
		warpPerspective(gray_frame, marker_image, M, Size(MARKER_SIZE, MARKER_SIZE));
		DumpImg(marker_image,"marker_imageXX" + to_string(i));
		//threshold(marker_image, marker_image, 125, 255, THRESH_BINARY | THRESH_OTSU); //OTSU determins threshold automatically.  

		/*
		//A marker must has a whole black border.  
		for (int y = 0; y < 7; ++y)// 8*MARKER_CELL_SIZE
		{
			int inc = (y == 0 || y == 6) ? 1 : 6;
			int cell_y = y*MARKER_CELL_SIZE;

			for (int x = 0; x < 7; x += inc)
			{
				int cell_x = x*MARKER_CELL_SIZE;
				int none_zero_count = countNonZero(marker_image(Rect(cell_x, cell_y, MARKER_CELL_SIZE, MARKER_CELL_SIZE)));
				if (none_zero_count > MARKER_CELL_SIZE*MARKER_CELL_SIZE / 4) //非黑像素少于1/4， 判断为黑色unit
					goto __wrongMarker;
			}
		}
		//comment: 这里的mark没有取向，长宽都是8个unit，内圈6*6 unit起编码作用
		//Decode the marker  
		for (int y = 0; y < 5; ++y)
		{
			int cell_y = (y + 1)*MARKER_CELL_SIZE;

			for (int x = 0; x < 5; ++x)
			{
				int cell_x = (x + 1)*MARKER_CELL_SIZE;
				int none_zero_count = countNonZero(marker_image(Rect(cell_x, cell_y, MARKER_CELL_SIZE, MARKER_CELL_SIZE)));
				if (none_zero_count > MARKER_CELL_SIZE*MARKER_CELL_SIZE / 2)
					bit_matrix.at<uchar>(y, x) = 1;
				else
					bit_matrix.at<uchar>(y, x) = 0;
			}
		}
		//to Do: hammingDistance(),bitMatrixToId(),bitMatrixRotate
		//有取向的marker不用rotate来实现解码？
		//Find the right marker orientation  
		bool good_marker = false;
		int rotation_idx;   //逆时针旋转的次数  
		for (rotation_idx = 0; rotation_idx < 4; ++rotation_idx)
		{
			if (hammingDistance(bit_matrix) == 0)
			{
				good_marker = true;
				break;
			}
			bit_matrix = bitMatrixRotate(bit_matrix);
		}
		if (!good_marker) goto __wrongMarker;

		//Store the final marker  
		Marker& final_marker = possible_markers[i];
		final_marker.m_id = bitMatrixToId(bit_matrix);
		std::rotate(final_marker.m_corners.begin(), final_marker.m_corners.begin() + rotation_idx, final_marker.m_corners.end());
		final_markers.push_back(final_marker);
		*/
	__wrongMarker:
		continue;
	}
}

void DisplayMarkerVec(Mat img, vector<Marker> markers ,double fps)
{
	for (int i = 0; i < markers.size(); i++) {
		markers[i].DisplayCorners(img);
	}
	putText(img, "FPS:"+to_string(fps), Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));       // 字体颜色
	imshow("MarkerRocog", img);

#ifdef _DEBUG_FILE_DUMP
	if (is_dump) {
		file_count++;
		string img_log_path = "D:\\Dump_Folder\\dump_" + file_time + "\\__" + to_string(file_count) + ".bmp";
		cv::imwrite(img_log_path, img);

		string file_log_path = "D:\\Dump_Folder\\dump_" + file_time + "\\__dump.txt";
		ofstream dump_file;
		dump_file.open(file_log_path, ios::out | ios::app);
		dump_file << "params: " << "thresh_kernel_R" << "(" <<block_size <<") " 
			<< "thresh_offset("<<offset_sub_constant << ") "
			<< "min_marker_size(" << min_marker_size << ") "
			<< "approx_ploy_epsilon(" << approx_poly_eps << ") "
			<< endl <<endl;
		dump_file.close();
	}
#endif // _DEBUG_FILE_DUMP

}

void DumpImg(const Mat &img, string img_name) 
{
	if (is_dump) {
		//file_count++;
		string dirstr = "D:\\Dump_Folder\\dump_" + file_time;
		const char *dir = dirstr.c_str();
		if(_access(dir,0)!=0){
			_mkdir(dir);
			cout << "success!"<<endl;
		}
#ifdef MORPH_
		string img_path = "D:\\Dump_Folder\\dump_" + file_time + "\\__" + to_string(file_count + 1) + img_name + ".bmp";
#else
		string img_path = "D:\\Dump_Folder\\dump_" + file_time + "\\__" + to_string(file_count + 1) + img_name + "NoMorph" + ".bmp";
#endif
		cv::imwrite(img_path, img);
	}
}