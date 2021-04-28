#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "getopt.h"
#include "trt.h"

using namespace std;

int main(int argc, char *argv[]){ //ARGV[1]
    int opt=0,option_index=0;
    static struct option opts[]={
        {"input-onnx",required_argument,nullptr,'i'},
        {"output-engine",required_argument,nullptr,'o'},
        {"mode",required_argument,nullptr,'m'},
        {"califile",required_argument,nullptr,'c'},
        {0,0,0,0}
    };

    std::string onnx = "test.onnx";
    std::string engine = "testFp16.engine";
    Tn::RUN_MODE mode = Tn::RUN_MODE::FLOAT16;
    std::string califile = "";
    while ((opt = getopt_long_only(argc,argv,"i:o:m:c:",opts,&option_index))!=-1){
        switch (opt)
        {
        case 'i':onnx=std::string(optarg);
            break;
        case 'o':engine=std::string(optarg);
            break;    
        case 'm':{int a=atoi(optarg);
            switch (a){
            case 0:mode=Tn::RUN_MODE::FLOAT32;
                break;
            case 1:mode=Tn::RUN_MODE::FLOAT16;
                break;
            case 2:mode=Tn::RUN_MODE::INT8;
                break;
            default:
                break;
            };    
            break;}
        case 'c':califile=std::string(optarg);
        default:
            break;
        }
    }

    std::cout<<"input-onnx "<<onnx<<std::endl
            <<"output-engine "<<engine<<std::endl;
    
    Tn::onnx2tensorrt net(onnx,1,califile,mode);
    net.saveEngine(engine);
    std::cout<<"save "<<engine<<std::endl;

    return 0;

}