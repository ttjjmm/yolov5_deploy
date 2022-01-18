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


struct object_rect {
    int x;
    int y;
    int width;
    int height;
};


int resize_uniform(cv::Mat& src, cv::Mat& dst, const cv::Size& dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    //std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        //std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        //std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
        exit(11);
    }
    //cv::imshow("dst", dst);
    //cv::waitKey(0);
    return 0;
}


int main() {
    cv::Mat img;
    std::string path2 = "/home/ubuntu/Documents/cpp/yolov5/dog.jpg";
    img = cv::imread(path2, cv::COLOR_BGR2RGB);
    if (img.empty()){
        fprintf(stderr, "cv::imread %s failed!", path2.c_str());
        return -1;
    }

    Yolov5 det("/home/ubuntu/Documents/cpp/yolov5/ncnn/yolov5s.param",
               "/home/ubuntu/Documents/cpp/yolov5/ncnn/yolov5s.bin",
               true);

    cv::Mat resize_img;
    auto input_size = det.input_size;
    object_rect eff{};
    resize_uniform(img, resize_img, input_size, eff);

    std::vector<BoxInfo> dets = det.detect(resize_img, 0.5, 0.5);

    for (auto& det_box: dets){
        auto pt1 = cv::Point2i(int(det_box.x1), int(det_box.y1));
        auto pt2 = cv::Point2i(int(det_box.x2), int(det_box.y2));
        cv::rectangle(resize_img, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
    }
    cv::imshow("res", resize_img);
    cv::waitKey(0);

    return 0;
}


