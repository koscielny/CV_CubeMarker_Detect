#include<opencv2\opencv.hpp>

using namespace cv;
using namespace std;
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;
const int BINARY_THRESHOLD = 230; //ԽСԽ����ȥ��ɫ����//threshold = 120: 8pm�չ�ƣ��ӽ�б����45��
const int BINARY_MAX = 255;

//enum CONTOUR_MODE_TYPE {
//	CV_RETR_EXTERNAL, CV_RETR_LIST
//};
//const CONTOUR_MODE_TYPE CONTOUR_MODE = CV_RETR_EXTERNAL;
//const int CONTOUR_MODE = CV_RETR_LIST;
const int CONTOUR_MODE = CV_RETR_LIST; //CV_RETR_EXTERNAL;//outermost:CV_RETR_EXTERNAL, all without hierarchy:CV_RETR_LIST
const int CONTOUR_METHOD = CV_CHAIN_APPROX_NONE;//none compress:CV_CHAIN_APPROX_NONE
Scalar CONTOUR_COLOR = Scalar(128, 255, 255);

const int THRESHOLD_TYPE = THRESH_BINARY;
int adaptive_method = ADAPTIVE_THRESH_GAUSSIAN_C; //ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C 	
int block_size = 17;//1 - 15
double offset_sub_constant = 5;//+/-/0

int gaussian_kernel_size = 5;
int border_type = BORDER_CONSTANT;		//BORDER_CONSTANT || BORDER_REPLICATE || BORDER_REFLECT ||
										//BORDER_WRAP || BORDER_REFLECT_101 || BORDER_TRANSPARENT || 
										//BORDER_REFLECT101 || BORDER_DEFAULT || BORDER_ISOLATED
int bilateral_kernel = 4;
double sigma_color = 120;
double sigma_space = 200;
int last_threshold = 230;

static int adaptive_method_trackingbar, block_size_trackingbar, constant_trackingbar, gaussian_kernel_trackingbar, border_type_trackingbar,
sigma_color_trackingbar, sigma_space_trackingbar, bilateral_kernel_trackingbar, last_threshold_trackingbar;

void onChange(int, void*);
/*
#pragma region MY_TEPLATE_OVERLOAD

template<typename _myTp> class myPoint_
{
public:
	//typedef _myTp value_type;

	myPoint_() : x(0), y(0) {};
	//myPoint_(_myTp _x, _myTp _y);
//	template<typename _Tp>
	myPoint_(_myTp _x, _myTp _y) : x(_x), y(_y) {};
	
	_myTp x, y; //< the point coordinates
public:
	template<typename T>
	myPoint_& operator = (const myPoint_<T>& pt);
	//myPoint_& operator = (const myPoint_& pt)
	//{
	//	x = pt.x; y = pt.y;
	//	return *this;
	//};
};

template<typename _myTp, typename T> inline
myPoint_<_myTp>& myPoint_<_myTp>::operator = (const myPoint_<T>& pt)
{
	x = pt.x; y = pt.y;
	return *this;
}

typedef myPoint_<int> myPoint2i;
typedef myPoint_<float> myPoint2f;
#pragma endregion

template <class SrcType, class DstType>
void ConvertVector2(const vector<SrcType>& src, vector<DstType>& dst) {
	dst.resize(src.size());
	std::copy(src.begin(), src.end(), dst.begin());
}
*/


int main(int argc, char* argv[])
{
	/*
	vector <myPoint2i> II{ myPoint2i(5, 3), myPoint2i(5,3), myPoint2i(5,3), myPoint2i(5,3) };
	vector <myPoint2f> FF{ myPoint2f(5.11111, 3.11111), myPoint2f(5.11111, 3.11111), myPoint2f(5.11111, 3.11111) };
	vector <myPoint2i> II2;
	vector <myPoint2f> FF2;
	ConvertVector2(FF, II2);
	for(int i = 0 ; i < II2.size(); i++)
		cout << II2[i].x << endl;
	ConvertVector2(II, FF2);
	for (int i = 0; i < II2.size(); i++)
		cout << FF2[i].x << endl;
		*/
	//������ͷ������Ƶ  
	VideoCapture videoCapture(0);

	if (!videoCapture.isOpened())
	{
		return -1;
	}

	//ѭ����ʾÿһ֡  
	videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);
	bool stop = false;
	namedWindow("GrayCamera");
	namedWindow("BinaryCamera");
	namedWindow("CannyCamera");
	namedWindow("ContourCamera");

	const int adaptive_method_count = 1, block_size_count = 15, constant_count = 1, gaussian_kernel_count = 15, border_type_count = 8, 
		sigma_C_count = 15, sigma_S_count = 15, bilateral_kernel_count = 9, last_threshold_count = 8;
	createTrackbar("�ֲ�����Ӧ����_adaptive_method_trackingbar",
		"BinaryCamera", &adaptive_method_trackingbar, adaptive_method_count, onChange, 0
	);//��˹/��ֵ
	createTrackbar("�ֲ���С_block_size_trackingbar",
		"BinaryCamera", &block_size_trackingbar, block_size_count, onChange, 0
	);
//	createTrackbar("ƫ��ֵ_constant_trackingbar",
//		"BinaryCamera", &constant_trackingbar, constant_count, onChange, 0
//	);

	createTrackbar("��ɫϵ��_sigma_color_trackingbar",
		"BinaryCamera", &sigma_color_trackingbar, sigma_C_count, onChange, 0
	);
	createTrackbar("����ϵ��_sigma_space_trackingbar",
		"BinaryCamera", &sigma_space_trackingbar, sigma_S_count, onChange, 0
	);
	createTrackbar("˫�ߺ˴�С_bilateral_kernel_trackingbar",
		"BinaryCamera", &bilateral_kernel_trackingbar, bilateral_kernel_count, onChange, 0
	);
	createTrackbar("����ֵ��ֵ��_last_threshold_trackingbar",
		"BinaryCamera", &last_threshold_trackingbar, last_threshold_count, onChange, 0
	);


//	createTrackbar("��˹�˴�С_gaussian_kernel_trackingbar",
//		"BinaryCamera", &gaussian_kernel_trackingbar, gaussian_kernel_count, onChange, 0
//	);
//	createTrackbar("�˲���Ե_border_type_trackingbar",
//		"BinaryCamera", &border_type_trackingbar, border_type_count, onChange, 0
//	);

	while (!stop)  
	{
		Mat bgr_frame, gray_frame, binary_frame, canny_frame, filtered_binary_frame, threshold_filtered_binary_frame, eq_filtered_binary_frame;//���ڴ洢����ͷ��ȡ��ÿһ֡ͼ��  
		//Mat contour_frame = Mat::zeros(DESIRED_CAMERA_WIDTH, DESIRED_CAMERA_HEIGHT, CV_8UC3);
		vector<vector<Point> > contour_points_0;
//BGR
		videoCapture >> bgr_frame;
		//		namedWindow("BGRCamera");
		//		imshow("BGRCamera", bgr_frame);//��ʾ��ǰ֡  
//Gray
		cvtColor(bgr_frame, gray_frame, CV_BGR2GRAY);
		imshow("GrayCamera", gray_frame);//��ʾ��ǰ֡  
//Binary
		adaptiveThreshold(gray_frame, binary_frame, BINARY_MAX, adaptive_method, THRESHOLD_TYPE, block_size, offset_sub_constant);
		//		threshold(gary_frame, binary_frame, BINARY_THRESHOLD, BINARY_MAX, CV_THRESH_BINARY); //��threshold��ֵ��
		//GaussianBlur(binary_frame, filtered_binary_frame, Size(0,0), gaussian_kernel_size, gaussian_kernel_size, border_type);//gaussian/ bilateral
		bilateralFilter(binary_frame, 
			filtered_binary_frame, 
			bilateral_kernel, 
			sigma_color, 
			sigma_space, 
			border_type
		);
		//filtered_binary_frame = binary_frame;
		imshow("BinaryCamera", filtered_binary_frame);//��ʾ��ǰ֡  
//Canny
		Canny(filtered_binary_frame, canny_frame, 0, 0);//, apertureSize = 3, L2gradient = false);//input:binary
		imshow("CannyCamera", canny_frame);//��ʾ��ǰ֡  
//Contour
		//equalizeHist(filtered_binary_frame, eq_filtered_binary_frame);//findContours�Զ�ֵΪ���룬���Ծ��⻯����ȻҪ��ֵ��
		//threshold(eq_filtered_binary_frame, threshold_filtered_binary_frame, 240, BINARY_MAX, CV_THRESH_BINARY);
		//imshow("eq_BinaryCamera", threshold_filtered_binary_frame);//��ʾ��ǰ֡  
		//threshold_filtered_binary_frame = filtered_binary_frame;
		threshold(filtered_binary_frame, threshold_filtered_binary_frame, last_threshold, BINARY_MAX, CV_THRESH_BINARY); //��ֵ�����ã��������������ֱ��ͼ���⻯
		//GaussianBlur(threshold_filtered_binary_frame, threshold_filtered_binary_frame, Size(0, 0), gaussian_kernel_size, gaussian_kernel_size, border_type);
		//equalizeHist(threshold_filtered_binary_frame, threshold_filtered_binary_frame);
		imshow("ThreshedFilteredBinaryCamera", threshold_filtered_binary_frame);//��ʾ��ǰ֡  
		findContours(threshold_filtered_binary_frame, contour_points_0, CONTOUR_MODE, CONTOUR_METHOD);//input:binary
		//findContours(canny_frame, contour_points_0, CONTOUR_MODE, CONTOUR_METHOD);//input:binary
		Mat contour_frame(bgr_frame.size(), CV_8UC3, Scalar(0));
		int levels = -1;
		drawContours(contour_frame, contour_points_0, levels, CONTOUR_COLOR);
//		GaussianBlur(contour_frame, contour_frame, Size(0, 0), gaussian_kernel_size, gaussian_kernel_size, border_type);
		imshow("ContourCamera", contour_frame);//��ʾ��ǰ֡  

		char key = waitKey(30);
		if (key == 27)
			break;
	}
	return 0;
}

void onChange(int boo, void* )
{
	//adaptive_method_trackingbar = ADAPTIVE_THRESH_GAUSSIAN_C; //ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_GAUSSIAN_C 	
	//block_size_trackingbar = 5;//1 - 15
	//constant_trackingbar = 0.0;//+/-/0
//adaptive_method_trackingbar
	switch (adaptive_method_trackingbar){
		case (0):
			adaptive_method = ADAPTIVE_THRESH_GAUSSIAN_C;
			break;
		case (1):
			adaptive_method = ADAPTIVE_THRESH_MEAN_C;
			break;
		default:			
			break;
	}
//block_size_trackingbar
	if (block_size_trackingbar <= 3)
		block_size = 3;
	else block_size = (block_size_trackingbar % 2) == 1 ? block_size_trackingbar : block_size_trackingbar - 1;
//constant_trackingbar
	offset_sub_constant = constant_trackingbar;
//gaussian_kernel_trackingbar
	if (gaussian_kernel_trackingbar <= 1)
		gaussian_kernel_size = 1;
	else gaussian_kernel_size = (gaussian_kernel_trackingbar % 2) == 1 ? gaussian_kernel_trackingbar : gaussian_kernel_trackingbar - 1;
//bilateral
	sigma_color = 30.0 * sigma_color_trackingbar;
	sigma_space = 30.0 * sigma_space_trackingbar;
	bilateral_kernel = bilateral_kernel_trackingbar >= 3 ? bilateral_kernel_trackingbar : 3;
	last_threshold = (100 + last_threshold_trackingbar * 20) < 250? (100 + last_threshold_trackingbar * 20) : 250;
//border_type
	switch (border_type_trackingbar) {			//BORDER_CONSTANT || BORDER_REPLICATE || BORDER_REFLECT ||
												//BORDER_WRAP || BORDER_REFLECT_101 || BORDER_TRANSPARENT || 
												//BORDER_REFLECT101 || BORDER_DEFAULT || BORDER_ISOLATED//��˹�˲�����ı߽�����//ûӰ��
	case (0):
		border_type = BORDER_CONSTANT;
		break;
	case (1):
		border_type = BORDER_REPLICATE;
		break;
	case (2):
		border_type = BORDER_REFLECT;
		break;
	case (3):
		border_type = BORDER_DEFAULT;
		break;
	case (4):
		border_type = BORDER_REFLECT_101;
		break;
	case (5):
		border_type = BORDER_DEFAULT;
		break;
	case (6):
		border_type = BORDER_REFLECT101;
		break;
	case (7):
		border_type = BORDER_DEFAULT;
		break;
	case (8):
		border_type = BORDER_ISOLATED;
		break;
	default:
		break;
	}
}