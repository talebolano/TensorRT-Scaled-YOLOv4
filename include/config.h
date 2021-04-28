#ifndef __CONFIG_H_
#define __CONFIG_H_

#include<numeric>
#include<string>


template<class T>
int getArraylen(T& array){
    return sizeof(array)/sizeof(array[0]);
}

const int max_per_img = 100;
const float vis_thresh=0.5;
const float nms_thresh=0.45;

const int inputsize[2] = {640,640};
const int num_anchors = 3; 
const int classes = 80;
const int yolo1[2] = {inputsize[0] / 32, inputsize[1] /32};
const int yolo2[2] = {inputsize[0] / 16, inputsize[1] /16};
const int yolo3[2] = {inputsize[0] / 8, inputsize[1] /8};
// if yolo model have 4 output(like yolov4-p6) add const int yolo4[2] = {inputsize[0] / 64, inputsize[1] /64};

const int yolo1_num = getArraylen(yolo1);
const int yolo2_num = getArraylen(yolo2);
const int yolo3_num = getArraylen(yolo3);
// if yolo model have 4 output(like yolov4-p6) add const int yolo4_num = getArraylen(yolo4);
const int yolo1_size = std::accumulate(yolo1,yolo1+yolo1_num,1,std::multiplies<int64_t>());
const int yolo2_size = std::accumulate(yolo2,yolo2+yolo2_num,1,std::multiplies<int64_t>());
const int yolo3_size = std::accumulate(yolo3,yolo3+yolo3_num,1,std::multiplies<int64_t>());
// if yolo model have 4 output(like yolov4-p6) add const int yolo4_size = std::accumulate(yolo4,yolo4+yolo4_num,1,std::multiplies<int64_t>());
const int yolo_size = num_anchors*(yolo1_size+yolo2_size+yolo3_size);
// if yolo model have 4 output(like yolov4-p6) change as const int yolo_size = num_anchors*(yolo1_size+yolo2_size+yolo3_size+yolo4_size);
const std::string input_name = "input";
const std::string output_names[3] = {"conf","cls","bbox"};

//如果改变类别，在这里更改
const std::string class_names[80] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus",
               "train", "truck", "boat", "traffic_light", "fire_hydrant",
               "stop_sign", "parking_meter", "bench", "bird", "cat", "dog",
               "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
               "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports_ball", "kite", "baseball_bat",
               "baseball_glove", "skateboard", "surfboard", "tennis_racket",
               "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl",
               "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
               "hot_dog", "pizza", "donut", "cake", "chair", "couch",
               "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop",
               "mouse", "remote", "keyboard", "cell_phone", "microwave",
               "oven", "toaster", "sink", "refrigerator", "book", "clock",
               "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush"};


#endif
