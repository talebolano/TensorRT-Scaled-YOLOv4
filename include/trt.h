#ifndef __TRT_NET_H_
#define __TRT_NET_H_

#include <string>
#include <iostream>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <vector>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <NvOnnxConfig.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <numeric>
#include <algorithm>
#include "resize.h"
#include "util.h"
#include <NvInferPlugin.h>
#include "post.h"
#include "EntroyCalibrator.h"



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
        //智能指针，
        using nvUniquePtr = unique_ptr<T,InferDeleter>;
    public:
        onnx2tensorrt(string &onnxfFile,int maxBatchSize,string &califilename, RUN_MODE mode=RUN_MODE::FLOAT16);
        onnx2tensorrt(string &enginfFile);
        ~onnx2tensorrt();
        onnx2tensorrt()=delete;
        //bool build();
        int infer_gpupost(const cv::Mat &img,float*conf,float*cls,float*bbox);

        void saveEngine(string& filename);
        //void loadEngine(std::string &enginfFile);
        vector<void *> mCudaBuffers;
        vector<size_t> mBindBufferSizes;
        cudaStream_t mCudaStream;
        void* mCudaImg;

    private:
        //samplesCommon::OnnxSampleParams mParams;
        nvinfer1::Dims mInputDims;
        //nvUniquePtr<nvinfer1::IExecutionContext> mContext;
        shared_ptr<nvinfer1::IExecutionContext> mContext;
        shared_ptr<nvinfer1::ICudaEngine> mEngine;
        //nvUniquePtr<nvinfer1::IRuntime> mRuntime;
        vector<float* > moutput;
        void initEngine();
};

}


#endif