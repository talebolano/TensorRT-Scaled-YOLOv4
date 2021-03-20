#ifndef _NMS_H_
#define _NMS_H_

#include <memory>
#include "config.h"
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

vector<vector<float>> nms(float*conf,float*cls,float*bbox, int ind_size);
void vis(cv::Mat &img,vector<vector<float>>result,string outputName,bool show,bool write);
void vis(cv::VideoWriter &videowriter,cv::Mat &img,vector<vector<float>>result,string outputName,bool show,bool write);

#endif
