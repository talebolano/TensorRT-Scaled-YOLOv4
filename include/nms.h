#ifndef _NMS_H_
#define _NMS_H_

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"


std::vector<std::vector<float>> nms(float*conf,float*cls,float*bbox, int ind_size);
void vis(cv::Mat &img,std::vector<std::vector<float>>result);
void plottrack(cv::Mat &img,std::vector<std::vector<float>>result);

#endif
