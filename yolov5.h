//
// Created by tjm on 2022/1/18.
//

#ifndef YOLOV5_YOLOV5_H
#define YOLOV5_YOLOV5_H

#include <opencv2/core/core.hpp>
#include <net.h>

struct YoloObject {
//    cv::Rect_<float> rect;
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};


struct CenterPrior
{
    int x;
    int y;
    int stride;
};


typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

namespace yolocv {
    typedef struct {
        int width;
        int height;
    } YoloSize;
}

typedef struct {
//    std::string name;
    int stride;
    std::vector<yolocv::YoloSize> anchors;
} YoloLayerData;


class Yolov5
{
public:
    Yolov5(const char* param, const char* bin, bool useGPU);
    ~Yolov5();

    static Yolov5* detector;
    ncnn::Net* Net;
    static bool hasGPU;
    // modify these parameters to the same with your config if you want to use your own model
    int input_size[2] = {416, 416}; // input height and width
    int num_class = 80; // number of classes. 80 for COCO
//    int reg_max = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = {8, 16, 32}; // strides of the multi-level feature.

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

    std::vector<YoloLayerData> anchors{
            {8,  {{10, 13}, {16, 30},  {33, 23}}},
            {16, {{30,  61}, {62,  45},  {59,  119}}},
            {32, {{116, 90}, {156, 198}, {373, 326}}},
    };


    static void preprocess(cv::Mat& image, ncnn::Mat& in);
    void decode_infer(ncnn::Mat& feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);

};









#endif //YOLOV5_YOLOV5_H
