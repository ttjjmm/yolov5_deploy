//
// Created by Jimmy Tao on 2022/1/18.
//

#include "yolov5.h"
#include <benchmark.h>

bool Yolov5::hasGPU = false;
Yolov5* Yolov5::detector = nullptr;

Yolov5::Yolov5(const char* param, const char* bin, bool useGPU)
{
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

Yolov5::~Yolov5()
{
    delete this->Net;
}


void Yolov5::preprocess(cv::Mat& image, ncnn::Mat& in)
{
    int img_w = image.cols;
    int img_h = image.rows;

    in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
    //in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, this->input_width, this->input_height);

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1./255.f, 1./255.f, 1./255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
}

BoxInfo Yolov5::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
//    for (int i = 0; i < 4; i++)
//    {
//        float dis = 0;
//        float* dis_after_sm = new float[this->reg_max + 1];
//        activation_function_softmax(dfl_det + i * (this->reg_max + 1), dis_after_sm, this->reg_max + 1);
//        for (int j = 0; j < this->reg_max + 1; j++)
//        {
//            dis += j * dis_after_sm[j];
//        }
//        dis *= stride;
//        //std::cout << "dis:" << dis << std::endl;
//        dis_pred[i] = dis;
//        delete[] dis_after_sm;
//    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size.width);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size.height);

    //std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}

void Yolov5::decode_infer(ncnn::Mat& feats,
                          std::vector<CenterPrior>& center_priors,
                          float threshold,
                          std::vector<std::vector<BoxInfo>>& results) {

    const int num_points = (int)center_priors.size();
    //printf("num_points:%d\n", num_points);
    int curr_stride = 32;
    int anchor_num = 3;
    std::vector<cv::Size_<int>> curr_anchors;
    //cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < num_points; idx++) {

        auto stride = center_priors[idx].stride;

        if (curr_stride != stride) {
            curr_stride = stride;
            anchor_num = (int)this->anchors.at(curr_stride).size();
            curr_anchors = this->anchors.at(curr_stride);
        }

        const int ct_x = center_priors[idx].x;
        const int ct_y = center_priors[idx].y;

        for (int anchor_i = 0; anchor_i < anchor_num; ++anchor_i) {

            const float* scores = feats.row(idx * anchor_num + anchor_i) + 5;
            float score = 0;
            int cur_label = 0;
            for (int label = 0; label < this->num_class; label++) {
                if (scores[label] > score) {
                    score = scores[label];
                    cur_label = label;
                }
            }
            if (score > threshold) {
                //std::cout << "label:" << cur_label << " score:" << score << std::endl;
                const float* bbox_pred = feats.row(idx);
                auto cx = (float)((bbox_pred[0] * 2 - 0.5 + ct_x) * curr_stride);
                auto cy = (float)((bbox_pred[1] * 2 - 0.5 + ct_y) * curr_stride);
                auto w = (float)(pow(bbox_pred[2] * 2, 2) * curr_anchors.at(anchor_i).width);
                auto h = (float)(pow(bbox_pred[3] * 2, 2) * curr_anchors.at(anchor_i).height);

                BoxInfo bbox;
                bbox.x1 = cx - w / 2;
                bbox.y1 = cy - h / 2;
                bbox.x2 = cx + w / 2;
                bbox.y2 = cy + h / 2;
                bbox.score = score * bbox_pred[4];
                bbox.label = cur_label;
                results[cur_label].push_back(bbox);
                //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
                //cv::imshow("debug", debug_heatmap);
            }
        }

    }
}


static void generate_grid_center_priors(const int input_height,
                                        const int input_width,
                                        std::vector<int>& strides,
                                        std::vector<CenterPrior>& center_priors) {
    for (int stride : strides)
    {
//        YoloLayerData anchor = anchors[i];
        int feat_w = ceil((float)input_width / (float)stride);
        int feat_h = ceil((float)input_height / (float)stride);

        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct{};
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
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

    std::vector<std::vector<BoxInfo>> results;
    results.resize(this->num_class);

    ncnn::Mat out;
    ex.extract("output", out);
    // printf("%d %d %d \n", out.w, out.h, out.c);

    // generate center priors in format of (x, y, stride)
    std::vector<CenterPrior> center_priors;
    generate_grid_center_priors(this->input_size.width,
                                this->input_size.height,
                                this->strides,
                                center_priors);

    std::cout << center_priors.size() << std::endl;
    std::cout << out.w << " x " << out.h << std::endl;

    this->decode_infer(out, center_priors, score_threshold, results);
    std::cout << results.size() << std::endl;

    std::vector<BoxInfo> dets;
    for (auto & result : results) {
        Yolov5::nms(result, nms_threshold);
        for (auto box : result) {
            dets.push_back(box);
        }
    }

    //double end = ncnn::get_current_time();
    //double time = end - start;
    //printf("Detect Time:%7.2f \n", time);
    std::cout <<  "dets: " << dets.size() << std::endl;
    return dets;
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

