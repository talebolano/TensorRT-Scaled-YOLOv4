#ifndef _ENTROY_CALIBRATOR_H_
#define _ENTROY_CALIBRATOR_H_

#include"NvInfer.h"
#include"config.h"
#include"util.h"
#include<vector>
#include<string>
#include<opencv2/opencv.hpp>

namespace nvinfer1{
class INt8EntroyCalibrator : public IInt8EntropyCalibrator2
{

public:

    INt8EntroyCalibrator(int batchsize,std::string caliimagelist, std::string filename,bool readCache=true);
    virtual ~INt8EntroyCalibrator();

    int getBatchSize() const override {return mbatchsize;}
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
    const void* readCalibrationCache(std::size_t& length) override;
    void writeCalibrationCache(const void* ptr, std::size_t length) override;
    std::vector<cv::Mat> get_cail_image(std::string &caliimagelist,int resizeh,int resizew);

private:
    std::string mfilename;

    std::vector<cv::Mat> mdata;
    int mbatchsize;

    int mCurBatchIdx;
    size_t mInputCount;
    bool mReadCache;
    void* mDeviceInput{nullptr};

    std::vector<char> mCailbrationCache;
};

}


#endif