#ifndef __TRT_NET_H_
#define __TRT_NET_H_

#include <string>
#include <vector>
#include <memory>
#include "NvInfer.h"
#include "opencv2/opencv.hpp"

namespace Tn{

    using namespace std;

    enum class RUN_MODE{
        FLOAT32 = 0,
        FLOAT16=1,
        INT8=2
    };
    struct InferDeleter{
        template <typename T>
        void operator()(T* obj) const{
            if (obj){
                obj->destroy();
            }
        }
    };


    class onnx2tensorrt{
        template <typename T>
        using nvUniquePtr = unique_ptr<T,InferDeleter>;
    public:
        onnx2tensorrt(string &onnxfFile,string &califilename, RUN_MODE mode=RUN_MODE::FLOAT16,int batchsize=1);
        onnx2tensorrt(string &enginfFile,int batchsize=1);
        ~onnx2tensorrt();
        onnx2tensorrt()=delete;
        vector<int> infer_gpupost(const vector<cv::Mat> &imgs,vector<vector<float>> &conf,vector<vector<float>> &cls,vector<vector<float>> &bbox);
        void saveEngine(string& filename);

    private:
        int mbatchsize;
        nvinfer1::Dims mInputDims;
        vector<void *> mCudaBuffers;
        vector<size_t> mBindBufferSizes;
        shared_ptr<nvinfer1::IExecutionContext> mContext;
        shared_ptr<nvinfer1::ICudaEngine> mEngine;
        vector<float* > moutput;
        void initEngine();
};

}


#endif