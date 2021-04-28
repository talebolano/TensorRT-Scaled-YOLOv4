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


    std::string engine = "test16.trt";
    std::string inputs = "test.jpg";
    std::string output = "result";
    int show=0;
    int save=0;

    int opt=0,option_index = 0;
    static struct option opts[]=
    {
        {"engine-model",required_argument,nullptr,'e'},
        {"inputs",required_argument,nullptr,'i'},
        {"output",required_argument,nullptr,'o'},
        {"show",no_argument,nullptr,'v'},
        {"save",no_argument,nullptr,'s'},
        {0,0,0,0}
    };

    while((opt=getopt_long_only(argc,argv,"e:i:o:vs",opts,&option_index))!=-1)
    {
        switch (opt)
        {
        case 'e':engine = std::string(optarg);break;
        case 'i':inputs = std::string(optarg);break;
        case 'o':output = std::string(optarg);break;
        case 'v':show = 1;break;
        case 's':save =1;break;
        
        default:
            break;
        }
    }


    Tn::onnx2tensorrt net(engine);

    float*conf=NULL;
    float*cls=NULL;
    float*bbox=NULL;
    conf = (float*)calloc(max_per_img,sizeof(float));
    cls = (float*)calloc(max_per_img*classes,sizeof(float));
    bbox = (float*)calloc(max_per_img*4,sizeof(float));
    int ind_size;

    if(inputs.find("jpg") != std::string::npos || inputs.find("png") != std::string::npos){

        std::cout<<"read image file "<<inputs<<std::endl;

        //for(int i=0;i<20;i++){
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat inputimage = cv::imread(inputs);
        cv::Mat rgb;
        cv::cvtColor(inputimage,rgb,CV_BGR2RGB);

        auto end_time = std::chrono::high_resolution_clock::now();
        float total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
        std::cout<<"process spend time "<<total<<" ms"<<std::endl;

        start_time = std::chrono::high_resolution_clock::now();

        ind_size = net.infer_gpupost(rgb,conf,cls,bbox);

        end_time = std::chrono::high_resolution_clock::now();
        total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
        std::cout<<"infer spend time "<<total<<" ms"<<std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        std::string output_name = output+".jpg";
        if(show){cv::namedWindow("output",1);}
        
        if(ind_size>0){
            vector<vector<float>> reslut = nms(conf,cls,bbox,ind_size);
            vis(inputimage,reslut);
        }

        if(save){
            cv::imwrite(output_name,inputimage);
        }
        if(show) {
            cv::imshow("output",inputimage);
		    cv::waitKey(0);
        }

        free(conf);
        free(cls);
        free(bbox);


        end_time = std::chrono::high_resolution_clock::now();
        total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
        std::cout<<"vis and nms spend time "<<total<<" ms"<<std::endl;
        cv::destroyAllWindows();
        //}
    }

    else if (inputs.find("mp4") != std::string::npos || inputs.find("avi") != std::string::npos)
    {   
        std::cout<<"read video from "<<inputs<<std::endl;
        std::string output_name = output+".avi";
        cv::VideoCapture cap;
        cap.open(inputs);
        if(!cap.isOpened()){
            std::cout<<"Error: video stream can't be opened!"<<std::endl;
            return 1;
        }        
        int h = cap.get(4);
        int w = cap.get(3);

        cv::VideoWriter writer(output_name,cv::VideoWriter::fourcc('X','V','I','D'),30,cv::Size(w,h));

        if(show){
            cv::namedWindow("output",0);

        }
        
        cv::Mat input_image;
        while (cap.read(input_image))
        {
            

            if(!input_image.data){
                continue;
            }
            auto start_time = std::chrono::high_resolution_clock::now();
            cv::Mat rgb;
            cv::cvtColor(input_image,rgb,CV_BGR2RGB);

            auto end_time = std::chrono::high_resolution_clock::now();
            float total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"process spend time "<<total<<" ms"<<std::endl;

            start_time = std::chrono::high_resolution_clock::now();
            ind_size = net.infer_gpupost(rgb,conf,cls,bbox);

            end_time = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"infer spend time "<<total<<" ms"<<std::endl;

            //后处理
            start_time = std::chrono::high_resolution_clock::now();
            
            if(ind_size>0){
                vector<vector<float>> reslut = nms(conf,cls,bbox,ind_size);
                vis(input_image,reslut);
            }

            if(show) {
                cv::imshow("output",input_image);
                cv::waitKey(5);
            }
            if(save){
                writer.write(input_image);
            }

            end_time = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"vis spend time "<<total<<" ms"<<std::endl;
            }

        free(conf);
        free(cls);
        free(bbox);
        cv::destroyAllWindows();
        cap.release();
        
    }
    else if (inputs.find("txt") != std::string::npos )
    {
        std::cout<<"read image list "<<inputs<<std::endl;
        std::ifstream inputImageNameList(inputs);
        std::list<std::string> fileNames;

        if(!inputImageNameList.is_open()){
            std::cout<<"can not read image list "<<inputs<<std::endl;
            return 1;
        }
        std::string strLine;
        while (std::getline(inputImageNameList,strLine))
        {
            fileNames.push_back(strLine);
        }
        inputImageNameList.close();

        for(auto it=fileNames.begin();it!=fileNames.end();++it){

            auto start_time = std::chrono::high_resolution_clock::now();
            cv::Mat inputimage = cv::imread(*it);
            cv::Mat rgb;
            cv::cvtColor(inputimage,rgb,CV_BGR2RGB);

            auto end_time = std::chrono::high_resolution_clock::now();
            float total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"process spend time "<<total<<" ms"<<std::endl;

            start_time = std::chrono::high_resolution_clock::now();
            ind_size = net.infer_gpupost(rgb,conf,cls,bbox);

            end_time = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"infer spend time "<<total<<" ms"<<std::endl;

            start_time = std::chrono::high_resolution_clock::now();

            std::string output_name = output+".jpg";
            if(show){cv::namedWindow("output",1);}
        
            if(ind_size>0){
                vector<vector<float>> reslut = nms(conf,cls,bbox,ind_size);
                vis(inputimage,reslut);
            }

            if(save){
                cv::imwrite(output_name,inputimage);
            }
            if(show) {
                cv::imshow("output",inputimage);
                cv::waitKey(0);
            }
                        

            end_time = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"vis spend time "<<total<<" ms"<<std::endl;

        }
        free(conf);
        free(cls);
        free(bbox);

        cv::destroyAllWindows();

    }
    
    else
    {
        std::cout<<"do not support this format,please use file end with .mp4/.avi/.jpg/.png"<<std::endl;
    }
    

    std::cout<<"success all"<< std::endl;

    return 0;

}

