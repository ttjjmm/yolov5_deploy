//
// Created by tjm on 2022/1/18.
//

#ifndef YOLOV5_YOLOV5_H
#define YOLOV5_YOLOV5_H

#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <net.h>


typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;


typedef struct LayerInfo {
    std::string name;
    int stride;
    std::vector<cv::Size_<int>> anchors;
} LayerInfo;


class Yolov5 {
public:
    Yolov5(const char* param, const char* bin, bool useGPU);
    ~Yolov5();

    static Yolov5* detector;
    ncnn::Net* Net;
    static bool hasGPU;
    cv::Size_<int> input_size = {416, 416}; // input height and width

    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);

    std::vector<std::string> labels{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                     "hair drier", "toothbrush" };
private:
    std::vector<LayerInfo> layer_info {
        // key: stride, value: anchors
            {"output", 8, {{10, 13}, {16, 30},  {33, 23}}},
            {"355", 16, {{30,  61}, {62,  45},  {59,  119}}},
            {"370", 32, {{116, 90}, {156, 198}, {373, 326}}},
    };
    void decode_layer(const ncnn::Mat& feats, const std::vector<cv::Size_<int>> &anchors,
                      int stride, float score_thr, std::vector<BoxInfo> &results) const;
    static void preprocess(cv::Mat& image, ncnn::Mat& in);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);

};









#endif //YOLOV5_YOLOV5_H
