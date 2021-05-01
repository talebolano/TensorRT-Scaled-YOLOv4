#ifndef _NMS_H_
#define _NMS_H_

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"


std::vector<std::vector<float>> nms(std::vector<float>conf,std::vector<float>cls,std::vector<float>bbox, int ind_size);
void vis(cv::Mat &img,std::vector<std::vector<float>>result);
void plottrack(cv::Mat &img,std::vector<std::vector<float>>result);

#endif
