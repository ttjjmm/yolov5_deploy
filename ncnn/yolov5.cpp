//
// Created by Jimmy Tao on 2022/1/18.
//

#include "yolov5.h"
#include <benchmark.h>

bool Yolov5::hasGPU = false;
Yolov5* Yolov5::detector = nullptr;


Yolov5::Yolov5(const char* param, const char* bin, bool useGPU) {
    this->Net = new ncnn::Net();
    // opt
#if NCNN_VULKAN
    Yolov5::hasGPU = ncnn::get_gpu_count() > 0;
#endif
    this->Net->opt.use_vulkan_compute = Yolov5::hasGPU && useGPU;
    this->Net->opt.use_fp16_arithmetic = true;
    this->Net->load_param(param);
    this->Net->load_model(bin);
}


Yolov5::~Yolov5() {
    delete this->Net;
}


void Yolov5::preprocess(cv::Mat& image, ncnn::Mat& in) {
    int img_w = image.cols;
    int img_h = image.rows;

    in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
    //in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, this->input_width, this->input_height);

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1./255.f, 1./255.f, 1./255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
}


void Yolov5::decode_layer(const ncnn::Mat& feat,
                          const std::vector<cv::Size_<int>> &anchors,
                          int stride,
                          const float score_thr,
                          std::vector<BoxInfo> &results) const {

    int grid_x = this->input_size.width / stride;
    int grid_y = this->input_size.height / stride;
    const int num_cls = feat.w - 5;
    const int num_anchor = (int)anchors.size();

    for (auto idx = 0; idx < num_anchor; ++idx) {
        const auto anchor_w = anchors[idx].width;
        const auto anchor_h = anchors[idx].height;
        const ncnn::Mat feat_anchor = feat.channel(idx);
        for (auto i = 0; i < grid_y; ++i) {
            for (auto j = 0; j < grid_x; ++j) {
                const float *pred_ptr = feat_anchor.row(i * grid_x + j);
                int label = 0;
                float pred_score = 0;

                for (auto label_i = 0; label_i < num_cls; label_i++) {
                    float score = pred_ptr[5 + label_i];
                    if (score > pred_score) {
                        label = label_i;
                        pred_score = score;
                    }
                }
                float confidence = pred_score * pred_ptr[4];
                if (confidence > score_thr) {
                    float cx = ((float)j + pred_ptr[0] * 2.f - 0.5f) * (float)stride;
                    float cy = ((float)i + pred_ptr[1] * 2.f - 0.5f) * (float)stride;
                    float pw = pow(pred_ptr[2] * 2.f, 2.f) * (float)anchor_w;
                    float ph = pow(pred_ptr[3] * 2.f, 2.f) * (float)anchor_h;

                    BoxInfo bbox;
                    bbox.x1 = cx - pw * 0.5f;
                    bbox.y1 = cy - ph * 0.5f;
                    bbox.x2 = cx + pw * 0.5f;
                    bbox.y2 = cy + ph * 0.5f;
                    bbox.score = confidence;
                    bbox.label = label;
                    results.push_back(bbox);
                }
            }
        }

    }
}


std::vector<BoxInfo> Yolov5::detect(cv::Mat image, float score_threshold, float nms_threshold)
{
    ncnn::Mat input;
    preprocess(image, input);

    //double start = ncnn::get_current_time();

    auto ex = this->Net->create_extractor();
    ex.set_light_mode(false);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    ex.set_vulkan_compute(Yolov5::hasGPU);
#endif
    ex.input("images", input);

    std::vector<BoxInfo> results;

    for (const auto &layer: this->layer_info) {
        ncnn::Mat blob;
        auto stride = layer.stride;
        auto anchors = layer.anchors;
        ex.extract(layer.name.c_str(), blob);
        std::vector<BoxInfo> dets;
        this->decode_layer(blob, anchors, stride, score_threshold, dets);
        results.insert(results.end(), dets.begin(), dets.end());
    }

//
    Yolov5::nms(results, nms_threshold);
    std::cout << results.size() << std::endl;
    return results;
}


void Yolov5::nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}

