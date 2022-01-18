//
// Created by tjm on 2022/1/18.
//

#include <iostream>
#include "yolov5.h"
//#include <vector>
//#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
//#include "benchmark.h"


int main() {
    // /home/tjm/Documents/python/pycharmProjects
    // /home/ubuntu/Documents/pycharm/
    cv::Mat img;
    std::string path2 = "/home/ubuntu/Documents/cpp/yolov5/dog.jpg";
    img = cv::imread(path2, cv::COLOR_BGR2RGB);
    if (img.empty()){
        fprintf(stderr, "cv::imread %s failed!", path2.c_str());
        return -1;
    }

//    ResizeImg(img, resize_img, 960, rat_a, rat_b);

    Yolov5 det("/home/ubuntu/Documents/cpp/yolov5/ncnn/yolov5s.param",
               "/home/ubuntu/Documents/cpp/yolov5/ncnn/yolov5s.bin",
               false);

    det.detect(img, 0.3, 0.5);

//    for (auto& det_box: res){
//        cv::rectangle(img, det_box.pt1, det_box.pt2, cv::Scalar(255, 0, 0), 2, cv::LINE_4);
//    }
    cv::imshow("res", img);
    cv::waitKey(0);

    return 0;
}


