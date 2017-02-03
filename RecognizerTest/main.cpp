#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <oe2.h>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    cv::namedWindow("input");
    cv::namedWindow("vis");
    cv::namedWindow("vis2");

    MarkRecognizer mark_recogniezer;
    mark_recogniezer.init("..");
    cv::Mat input_frame;
    while (true) {
        cap >> input_frame;
        imshow("input", input_frame);
        mark_recogniezer.ProcessRGBFrame(input_frame);
        int keycode = cv::waitKey(1);
        if (keycode == 'q')
            break;
    }
}