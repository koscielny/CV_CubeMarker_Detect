#ifndef DEBUGTOOL_HPP
#define DEBUGTOOL_HPP

void initUI(void)
{
	namedWindow("GrayCamera");
	namedWindow("BinaryCamera");
	namedWindow("ContourCamera");
	namedWindow("MarkerRocog");
	namedWindow("Toolbox", WINDOW_NORMAL);

	createTrackbar("adaptive_method_trackingbar", "Toolbox", 
		&adaptive_method_trackingbar, adaptive_method_count, onChange, 0);
	createTrackbar("block_size_trackingbar", "Toolbox", 
		&block_size_trackingbar, block_size_count, onChange, 0);
	createTrackbar("constant_trackingbar", "Toolbox", 
		&constant_trackingbar, constant_count, onChange, 0);
	//createTrackbar("8*Mark´óÐ¡_marker_min_size_trackingbar", "BinaryCamera",//ÍÆ¼öÖµ ¾¡Á¿´óÓÚ15*8//ÁíÒ»ÖÖËµ·¨Ô¼µÈÓÚ8 * block_size_trackbar = 2 * block_size 
	//	&marker_min_size_trackingbar, marker_min_size_count, onChange, 0);
	createTrackbar("0.1*EPS_approx_poly_eps_trackingbar", "Toolbox",
		&approx_poly_eps_trackingbar, approx_poly_eps_count, onChange, 0);
	createTrackbar("50*AREA_min_area_trackingbar", "Toolbox",
		&min_area_trackingbar, min_area_count, onChange, 0);
	createTrackbar("sigma_color_trackingbar",
		"Toolbox", &sigma_color_trackingbar, sigma_C_count, onChange, 0
		);
	createTrackbar("sigma_space_trackingbar",
		"Toolbox", &sigma_space_trackingbar, sigma_S_count, onChange, 0
		);
	createTrackbar("bilateral_kernel_trackingbar",
		"Toolbox", &bilateral_kernel_trackingbar, bilateral_kernel_count, onChange, 0
		);
}

void onChange(int boo, void*)
{
	//adaptive_method_trackingbar
	switch (adaptive_method_trackingbar) {
	case (0) :
		adaptive_method = ADAPTIVE_THRESH_GAUSSIAN_C;
		break;
	case (1) :
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
#endif