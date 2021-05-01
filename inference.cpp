#include "getopt.h"
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "config.h"
#include "nms.h"
#include "trt.h"

using namespace std;

int main(int argc, char* argv[]){


    string engine = "test16.trt";
    string inputs = "test.jpg";
    string output = "result";
    int show=0;
    int save=0;
    int batchsize = 1;

    int opt=0,option_index = 0;
    static struct option opts[]=
    {
        {"engine-model",required_argument,nullptr,'e'},
        {"inputs",required_argument,nullptr,'i'},
        {"output",required_argument,nullptr,'o'},
        {"show",no_argument,nullptr,'v'},
        {"save",no_argument,nullptr,'s'},
        {"batchsize",required_argument,nullptr,'b'},
        {0,0,0,0}
    };

    while((opt=getopt_long_only(argc,argv,"e:i:o:b:vs",opts,&option_index))!=-1)
    {
        switch (opt)
        {
        case 'e':engine = string(optarg);break;
        case 'i':inputs = string(optarg);break;
        case 'o':output = string(optarg);break;
        case 'v':show = 1;break;
        case 's':save =1;break;
        case 'b':batchsize = atoi(optarg);break;
        
        default:
            break;
        }
    }


    Tn::onnx2tensorrt net(engine,batchsize);

    vector<vector<float>> conf;
    vector<vector<float>> cls;
    vector<vector<float>> bbox;
    vector<int> ind_size;


    if (inputs.find("mp4") != string::npos || inputs.find("avi") != string::npos)
    {   
        cout<<"read video from "<<inputs<<endl;
        string output_name = output+".avi";
        cv::VideoCapture cap;
        cap.open(inputs);
        if(!cap.isOpened()){
            cout<<"Error: video stream can't be opened!"<<endl;
            return 1;
        }        
        int h = cap.get(4);
        int w = cap.get(3);
        int fps = cap.get(CV_CAP_PROP_FPS);

        cv::VideoWriter writer(output_name,cv::VideoWriter::fourcc('X','V','I','D'),fps,cv::Size(w,h));

        if(show){
            cv::namedWindow("output",0);

        }
        
        cv::Mat input_image;
        bool ret = true;
        while (ret)
        {
            vector<cv::Mat> inputImgs;

            for(int b=0;b<batchsize;++b){
                ret = cap.read(input_image);
                if(ret){
                    inputImgs.push_back(input_image);
                }
                else{
                    break;
                }
                
            }

            ind_size = net.infer_gpupost(inputImgs,conf,cls,bbox);
            
            for(int b=0;b<batchsize;++b){
                string output_name = output+".jpg";
                vector<vector<float>> reslut = nms(conf[b],cls[b],bbox[b],ind_size[b]);
                vis(inputImgs[b],reslut);
                if(save){
                    cv::imwrite(output_name,inputImgs[b]);
                    }
                if(show) {
                    cv::imshow("output",inputImgs[b]);
                    cv::waitKey(0);
                }

            }


        }

        cv::destroyAllWindows();
        cap.release();
        
    }
    else if (inputs.find("txt") != string::npos ){
        cout<<"read image list "<<inputs<<endl;
        ifstream inputImageNameList(inputs);
        vector<string> fileNames;

        if(!inputImageNameList.is_open()){
            cout<<"can not read image list "<<inputs<<endl;
            return 1;
        }
        string strLine;
        while (getline(inputImageNameList,strLine))
        {
            fileNames.push_back(strLine);
        }
        inputImageNameList.close();

        int imageNum = fileNames.size();
        int epochs = imageNum / batchsize + 1;
        int lastbatchsize = imageNum % batchsize;
        int nowbatchsize;
        if(show){
            cv::namedWindow("output",0);

        }
        for(int e=0;e<epochs;++e){
            vector<cv::Mat> inputImgs;
            if(e==epochs-1){
                nowbatchsize == lastbatchsize;
            }

            for(int b=0;b<nowbatchsize;++b){
                cv::Mat inputimage = cv::imread(fileNames[e*batchsize+b]);
                inputImgs.push_back(inputimage);

            }
            ind_size = net.infer_gpupost(inputImgs,conf,cls,bbox);

            for(int b=0;b<nowbatchsize;++b){
                string output_name = output+".jpg";
                vector<vector<float>> reslut = nms(conf[b],cls[b],bbox[b],ind_size[b]);
                vis(inputImgs[b],reslut);
                if(save){
                    cv::imwrite(output_name,inputImgs[b]);
                    }
                if(show) {
                    cv::imshow("output",inputImgs[b]);
                    cv::waitKey(0);
                }

            }

        }
        cv::destroyAllWindows();

    }
    
    else
    {
        cout<<"do not support this format,please use file end with .mp4/.avi/.jpg/.png"<<endl;
    }


    cout<<"success all"<< endl;

    return 0;

}

