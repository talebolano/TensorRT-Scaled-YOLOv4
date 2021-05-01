#include <opencv2/opencv.hpp>
#include <vector>
#include "config.h"
#include "resize.h"

using namespace std;

vector<float> processing(const vector<cv::Mat> &imgs,bool keepration ,bool keepcenter){

    // input : bhwc uchar  output:bchw float

    const int batchsize = imgs.size();

    vector<float> inputData(batchsize*3*inputsize[1]*inputsize[0]);

    auto dataptr = inputData.data();

    for(int i=0;i<batchsize;++i){
        cv::Size oriSize = imgs[i].size();
        cv::Mat rgb;
        cv::cvtColor(imgs[i],rgb,CV_BGR2RGB);

        float scaleX = (oriSize.width*1.0f / inputsize[1]);
        float scaleY = (oriSize.height*1.0f / inputsize[0]);
        float shiftX = 0.f ,shiftY = 0.f;
        if(keepration)scaleX = scaleY = scaleX > scaleY ? scaleX : scaleY;
        if(keepration && keepcenter){shiftX = (inputsize[1] - oriSize.width/scaleX)/2.f;shiftY = (inputsize[0] - oriSize.height/scaleY)/2.f;}

        cv::Mat resizedimg;

        float inputH = oriSize.height / scaleY;
        float inputW = oriSize.width / scaleX;

        cv::resize(rgb,resizedimg,cv::Size(inputW,inputH));

        cv::Mat inputFloat;

        resizedimg.convertTo(inputFloat,CV_32FC3,1/255.);

        cv::Mat inputPad(cv::Size(inputsize[1],inputsize[0]),CV_32FC3);

        inputFloat.copyTo(inputPad(cv::Rect(shiftX,shiftY,inputW,inputH)));

        vector<cv::Mat> input_channels(3);

        cv::split(inputPad,input_channels);

        for(int j=0;j<3;++j){
            memcpy(dataptr,input_channels[i].data,inputsize[0]*inputsize[1]*sizeof(float));
            dataptr += inputsize[0]*inputsize[1];

        }
    }
    return inputData;

}
