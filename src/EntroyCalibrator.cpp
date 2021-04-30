#include "EntroyCalibrator.h"
#include "cuda.h"
#include "config.h"
#include "util.h"
#include "resize.h"
#include "cuda_runtime.h"
#include <iostream>
#include <fstream>
#include <iterator>

namespace nvinfer1{
    INt8EntroyCalibrator::INt8EntroyCalibrator(int batchsize,std::string caliimagelist, std::string filename,bool readCache)
        :mfilename(filename),mbatchsize(batchsize),mReadCache(readCache){
        
        mdata = get_cail_image(caliimagelist,inputsize[1],inputsize[0]);//resize 

        mInputCount = mbatchsize*inputsize[0]*inputsize[1]*3;// b*c*h*w

        mCurBatchIdx = 0;
        CHECK(cudaMalloc(&mDeviceInput,mInputCount*sizeof(float) )) ;

    }
    INt8EntroyCalibrator::~INt8EntroyCalibrator(){
        cudaFree(mDeviceInput);
    }


    bool INt8EntroyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings){
        if(mCurBatchIdx+mbatchsize >int(mdata.size())){
            return false;
        }

        void* g_temp_img;
        CHECK(cudaMalloc(&g_temp_img,inputsize[0]*inputsize[1]*3*sizeof(uchar)));

        for(auto iter = mdata.begin()+mCurBatchIdx;iter!=mdata.begin()+mCurBatchIdx+mbatchsize;++iter){
            int i = 0 ;
            
            CHECK(cudaMemcpy(g_temp_img,iter->data,mbatchsize*inputsize[0]*inputsize[1]*3*sizeof(uchar),cudaMemcpyHostToDevice)); // uchar hwc -> float chw

            resizeAndNorm(g_temp_img,(float*)mDeviceInput+i*mbatchsize*inputsize[0]*inputsize[1]*3,iter->cols,iter->rows,inputsize[0],inputsize[1],0,0);
            cudaDeviceSynchronize();
            i +=1;
        }
        CHECK(cudaFree(g_temp_img));
        bindings[0] = mDeviceInput;
        
        std::cout<<"load batch "<<mCurBatchIdx<<" to "<<mCurBatchIdx+mbatchsize-1<<std::endl;
        mCurBatchIdx +=mbatchsize;
        return true;

    }

    const void* INt8EntroyCalibrator::readCalibrationCache(std::size_t& length){
        mCailbrationCache.clear();
        std::ifstream input(mfilename,std::ios::binary);
        input>>std::noskipws;
        if(mReadCache & input.good()){
            std::copy(std::istream_iterator<char>(input),std::istream_iterator<char>(),std::back_inserter(mCailbrationCache));

        }
        length = mCailbrationCache.size();
        return length ? &mCailbrationCache[0] :nullptr;

    }

    void INt8EntroyCalibrator::writeCalibrationCache(const void* ptr, std::size_t length){
        std::ofstream output(mfilename,std::ios::binary);
        output.write(reinterpret_cast<const char*>(ptr),length);
    }

    std::vector<cv::Mat> INt8EntroyCalibrator::get_cail_image(std::string &caliimagelist,int resizeh,int resizew){
        std::ifstream file(caliimagelist);
        assert(file.is_open());
        std::vector<cv::Mat> outimg;
        std::string strLine;
        std::cout<<"begin read cali image"<<std::endl;
        while (std::getline(file,strLine))
        {
            cv::Mat image = cv::imread(strLine);

            cv::Mat reiszed(resizeh,resizew,CV_8UC3);// resize

            cv::resize(image,reiszed,reiszed.size());
            
            cv::Mat rgb;
            cv::cvtColor(reiszed,rgb,cv::COLOR_BGR2RGB);

            outimg.push_back(rgb);
        }
        std::cout<<"read cali image over"<<std::endl;
        file.close();

        return outimg;
    }


}