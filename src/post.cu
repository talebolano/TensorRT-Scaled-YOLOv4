#include "post.h"
#include "config.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/sort.h"
#include "thrust/count.h"
#include "thrust/sequence.h"
#include "thrust/execution_policy.h"

struct is_morethan_thresh
{
    __host__ __device__
    bool operator()(float &x){
        return x>vis_thresh; //为了获得更大的速度，可以在这里设为0.3，因为之后都是对 cate的衰减
    }
};

__global__ void postKernel(float* cls,float* bbox,int*g_temp_index,float*out_cls,float*out_bbox,int ind_size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>ind_size-1) return;

    for(int c=0;c<classes;++c){
        out_cls[idx*classes+c] = cls[g_temp_index[idx]*classes+c];
    }

    for(int i=0;i<4;++i){
        out_bbox[idx*4+i] = bbox[g_temp_index[idx]*4+i];
    }

}



int post_gpu(float*conf,float*cls,float*bbox,float*out_conf,
             float*out_cls,float*out_bbox){
    /*
    排序后去除小于阈值的，并对 cls 和 bbox 进行排序
    并拷贝到cpu conf cls bbox
    */
    

    int*g_temp_index;
    int ind_size;

    cudaMalloc((void**)&g_temp_index,yolo_size*sizeof(int));

    thrust::sequence(thrust::device,g_temp_index,g_temp_index+yolo_size);
    
    thrust::stable_sort_by_key(thrust::device,conf,conf+yolo_size,g_temp_index,thrust::greater<float>());//根据cate_score排序index
    cudaDeviceSynchronize();
    ind_size = thrust::count_if(thrust::device,conf,conf+yolo_size,is_morethan_thresh());//去除cate_scores中小于阈值的
    cudaDeviceSynchronize();
    if (ind_size>max_per_img){
        ind_size=max_per_img;
    }else if (ind_size==0)
    {
        cudaFree(g_temp_index);
        return 0;
    }
    float *out_cls_gpu;
    float *out_bbox_gpu; 
    cudaMalloc((void**)&out_cls_gpu,ind_size*classes*sizeof(float));
    cudaMalloc((void**)&out_bbox_gpu,ind_size*4*sizeof(float));
    

    const int blockSize = 512;
    const int gridSize = (ind_size + blockSize - 1) / blockSize;


    postKernel<<<gridSize,blockSize>>>(cls,bbox,g_temp_index,out_cls_gpu,out_bbox_gpu,ind_size);
    cudaDeviceSynchronize();

    cudaMemcpy(out_conf,conf,ind_size*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(out_cls,out_cls_gpu,ind_size*classes*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(out_bbox,out_bbox_gpu,ind_size*4*sizeof(float),cudaMemcpyDeviceToHost);
    
    
    cudaFree(g_temp_index);
    cudaFree(out_cls_gpu);
    cudaFree(out_bbox_gpu);

    return ind_size;

}