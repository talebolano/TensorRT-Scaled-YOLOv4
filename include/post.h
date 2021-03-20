#ifndef TN_POST_H_
#define TN_POST_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <memory>
#include "config.h"

int post_gpu(float*conf,float*cls,float*bbox,float*out_conf,
             float*out_cls,float*out_bbox   
                );
#endif