#ifndef TN_RESIZE_H_
#define TN_RESIZE_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef unsigned char uchar;

int resizeAndNorm(void*p,float*d,int w,int h,int in_w,int in_h,bool keepraton,bool keepcenter);
#endif
