#ifndef TN_POST_H_
#define TN_POST_H_

int post_gpu(float*conf,float*cls,float*bbox,float*out_conf,
             float*out_cls,float*out_bbox   
                );
#endif