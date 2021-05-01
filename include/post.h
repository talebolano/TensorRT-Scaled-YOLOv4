#ifndef TN_POST_H_
#define TN_POST_H_

#include <vector>

std::vector<int> post_gpu(const int batchsize,float*conf,float*cls,float*bbox,
            std::vector<std::vector<float>>out_conf,std::vector<std::vector<float>>out_cls,std::vector<std::vector<float>>out_bbox);
#endif