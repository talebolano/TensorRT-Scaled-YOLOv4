#ifndef TN_RESIZE_H_
#define TN_RESIZE_H_

#include <vector>
#include <opencv2/opencv.hpp>

typedef unsigned char uchar;

std::vector<float> processing(const std::vector<cv::Mat> &imgs,bool keepration ,bool keepcenter);
int resizeAndNorm(void * p,float *d,int w,int h,int in_w,int in_h, bool keepration ,bool keepcenter);
#endif
