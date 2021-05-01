#ifndef TN_RESIZE_H_
#define TN_RESIZE_H_

#include <vector>

typedef unsigned char uchar;

std::vector<float> processing(const std::vector<cv::Mat> &imgs,bool keepration ,bool keepcenter);
#endif
