#include "trt.h"
#include "mish.h"
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <ostream>
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <numeric>
#include <algorithm>
#include "resize.h"
#include "util.h"
#include "NvInferPlugin.h"
#include "post.h"
#include "EntroyCalibrator.h"
#include "cuda_runtime.h"
#include "config.h"


static Logger gLogger;                                                                           

namespace Tn{

    inline unsigned int getElementSize(nvinfer1::DataType t)
    {
        switch (t)
        {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        }
        throw std::runtime_error("Invalid DataType.");
        return 0;
    }
    inline int64_t volume(const nvinfer1::Dims& d)
    {
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }

    
    inline void* safeCudaMalloc(size_t memSize)
    {
        void* deviceMem;
        CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr){
            std::cerr << "Out of memory" << std::endl;
            exit(1);
        }
        return deviceMem;
    }

    onnx2tensorrt::onnx2tensorrt(std::string &onnxfFile,std::string &califilename,Tn::RUN_MODE mode,int batchsize):mbatchsize(batchsize){
        cudaSetDevice(0);
        initMishPlugin();
        auto builder = nvUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
        if(!builder){
            exit(EXIT_FAILURE);
        }

        const auto explicitBatch = 1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
        auto network = nvUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if(!network){
            exit(EXIT_FAILURE);
        }
        auto config = nvUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if(!config){
            exit(EXIT_FAILURE);
        }
        auto parser = nvUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network,gLogger.getTRTLogger()));
        if(!parser){
            exit(EXIT_FAILURE);
        }
        auto parsed = parser->parseFromFile(
            onnxfFile.c_str(),
            static_cast<int>(gLogger.getReportableSeverity())
        );
        if(!parsed){
            std::cerr<<"failed to parse from "<<onnxfFile<<"!"<<std::endl;
            exit(EXIT_FAILURE);
        }
        auto input = network->getInput(0);
        auto profileCalib = builder->createOptimizationProfile();
        profileCalib->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, inputsize[0], inputsize[1]});
        profileCalib->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{mbatchsize, 3, inputsize[0], inputsize[1]});
        profileCalib->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{mbatchsize, 3, inputsize[0], inputsize[1]});
        config->setCalibrationProfile(profileCalib);

        //builder->setMaxBatchSize(maxBatchSize);
        config->setMaxWorkspaceSize(1<<30);

        nvinfer1::INt8EntroyCalibrator *calibrator = nullptr;

        if(califilename.size()>0)  calibrator = new nvinfer1::INt8EntroyCalibrator(1,califilename,std::string("calib.table"));

        if(mode==RUN_MODE::INT8){
            std::cout<<"INT8 mode "<<std::endl;
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            config->setInt8Calibrator(calibrator);
        }
        else if(mode==RUN_MODE::FLOAT16){
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout<<"FP16 mode "<<std::endl;
        }
        else std::cout<<"FP32 mode "<<std::endl;

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network,*config),InferDeleter());


        if(!mEngine)
        {
            exit(EXIT_FAILURE);
        }
        if(calibrator){
            delete calibrator;
            calibrator = nullptr;
        }
        assert(network->getNbInputs()==1);
        mInputDims = network->getInput(0)->getDimensions();

        assert(mInputDims.nbDims==4);

        assert(network->getNbOutputs()==3);
        std::cout<<"haved genrated engine"<<std::endl;
        initEngine();
        }

    onnx2tensorrt::onnx2tensorrt(std::string &enginfFile,int batchsize):mbatchsize(batchsize){

        cudaSetDevice(0);
        initMishPlugin();
        std::fstream file;
        file.open(enginfFile,std::ios::binary|std::ios::in);
        if(!file.is_open()){
            std::cerr<<"read engine "<<enginfFile<<" failed"<<std::endl;
            return;
        }
        file.seekg(0,ios::end);
        int length = file.tellg();
        file.seekg(0,std::ios::beg);
        unique_ptr<char[]> data(new char[length]);
        file.read(data.get(),length);
        file.close();

        initLibNvInferPlugins(&gLogger,"");
        auto mRuntime = nvUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
        if(!mRuntime)
        {
            exit(EXIT_FAILURE);
        }
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(data.get(),length),InferDeleter());
        if(!mEngine)
        {
            exit(EXIT_FAILURE);
        }
        
        initEngine();
        std::cout<<"load genrated engine"<<std::endl;

    }


    void onnx2tensorrt::saveEngine(std::string& filename){
        if(mEngine)
        {
            shared_ptr<nvinfer1::IHostMemory> data(mEngine->serialize(),InferDeleter());
            std::ofstream file;
            file.open(filename,std::ios::binary | std::ios::out);
            if(!file.is_open())
            {
                std::cerr<<"read create file failed"<<std::endl;
                file.close();
            }
            file.write((const char *)data->data(),data->size());
            file.close();
        }
        return;
    }

    void onnx2tensorrt::initEngine(){
        mContext = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext(),InferDeleter());

        assert(mContext!=nullptr);
        int nbBindings = mEngine->getNbBindings();

        mCudaBuffers.resize(nbBindings);
        mBindBufferSizes.resize(nbBindings); 

        for(int i=0;i<nbBindings;++i){

            nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
            int totalSize = abs(mbatchsize*volume(mEngine->getBindingDimensions(i))*getElementSize(dtype));//
            assert(totalSize>0);
            mBindBufferSizes[i] = totalSize;
            mCudaBuffers[i] = safeCudaMalloc(totalSize);   
        }

        mInputDims = mEngine->getBindingDimensions(0);

        moutput.resize(mBindBufferSizes.size()-1);
        for(int i = 0;i<mBindBufferSizes.size()-1;++i){
            int index = mEngine->getBindingIndex(output_names[i].c_str());
            moutput[i] = (float*)mCudaBuffers[index];
        }

        return;
    }

    vector<int> onnx2tensorrt::infer_gpupost(const vector<cv::Mat> &imgs,vector<vector<float>> &conf,vector<vector<float>> &cls,vector<vector<float>> &bbox){
        bool keepRation=1,keepCenter=1;
        const int batchsize = imgs.size();
        mInputDims.d[0] = batchsize;
        mContext->setBindingDimensions(0,mInputDims);

        vector<float> inputData = processing(imgs,keepRation,keepCenter); // bhwc -->bchw
        CHECK(cudaMemcpy(mCudaBuffers[0],inputData.data(),inputData.size()*sizeof(float),cudaMemcpyHostToDevice));

        //resizeAndNorm(mCudaImg,(float*)mCudaBuffers[0],img.cols,img.rows,mInputDims.d[3],mInputDims.d[2],keepRation,keepCenter);
        mContext->executeV2(&mCudaBuffers[0]);
        cudaDeviceSynchronize();
        vector<int> ind_size = post_gpu(batchsize,(float*)moutput[0],(float*)moutput[1],(float*)moutput[2],
                            conf,cls,bbox);

        return ind_size;

    }

    onnx2tensorrt::~onnx2tensorrt(){

        mContext.reset();
        for(size_t bindindIdx=0;bindindIdx<mBindBufferSizes.size();++bindindIdx){
            if(mCudaBuffers[bindindIdx])CHECK(cudaFree(mCudaBuffers[bindindIdx]));

        }
        
    }




}
