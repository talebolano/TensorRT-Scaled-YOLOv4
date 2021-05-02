#include "post.h"
#include "config.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/sort.h"
#include "thrust/count.h"
#include "thrust/sequence.h"
#include "thrust/execution_policy.h"
#include "thrust/device_vector.h"

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


using namespace std;
vector<int> post_gpu(const int batchsize,float*conf,float*cls,float*bbox,
            vector<vector<float>> &out_conf,vector<vector<float>> &out_cls,vector<vector<float>> &out_bbox){
    /*
    排序后去除小于阈值的，并对 cls 和 bbox 进行排序
    并拷贝到cpu conf cls bbox
    */
    

    int*g_temp_index;
    vector<int> ind_size;
    ind_size.resize(batchsize);
    out_conf.resize(batchsize);
    out_cls.resize(batchsize);
    out_bbox.resize(batchsize);

    cudaMalloc((void**)&g_temp_index,yolo_size*sizeof(int));
    
    for(int i=0;i<batchsize;++i){
        thrust::sequence(thrust::device,g_temp_index,g_temp_index+yolo_size);
        
        thrust::stable_sort_by_key(thrust::device,conf+batchsize*yolo_size,conf+(batchsize+1)*yolo_size,g_temp_index,thrust::greater<float>());//根据cate_score排序index
        cudaDeviceSynchronize();
        ind_size[i] = thrust::count_if(thrust::device,conf+batchsize*yolo_size,conf+(batchsize+1)*yolo_size,is_morethan_thresh());//去除cate_scores中小于阈值的
        cudaDeviceSynchronize();
        if (ind_size[i]>max_per_img){
            ind_size[i]=max_per_img;
        }

        if(ind_size[i]>0){
            float *out_cls_gpu;
            float *out_bbox_gpu; 
            cudaMalloc((void**)&out_cls_gpu,ind_size[i]*classes*sizeof(float));
            cudaMalloc((void**)&out_bbox_gpu,ind_size[i]*4*sizeof(float));
            

            const int blockSize = 512;
            const int gridSize = (ind_size[i] + blockSize - 1) / blockSize;


            postKernel<<<gridSize,blockSize>>>(cls+classes*batchsize*yolo_size,bbox+4*batchsize*yolo_size,
                                        g_temp_index,out_cls_gpu,out_bbox_gpu,ind_size);
            cudaDeviceSynchronize();

            cudaMemcpy(out_conf[i].data(),conf,ind_size[i]*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(out_cls[i].data(),out_cls_gpu,ind_size[i]*classes*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(out_bbox[i].data(),out_bbox_gpu,ind_size[i]*4*sizeof(float),cudaMemcpyDeviceToHost);
            
            
            cudaFree(g_temp_index);
            cudaFree(out_cls_gpu);
            cudaFree(out_bbox_gpu);
        }

    }
    return ind_size;

}