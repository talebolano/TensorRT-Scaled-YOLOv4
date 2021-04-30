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
        {"batch-size",required_argument,nullptr,'b'},
        {0,0,0,0}
    };

    string onnx = "test.onnx";
    string engine = "testFp16.engine";
    Tn::RUN_MODE mode = Tn::RUN_MODE::FLOAT16;
    string califile = "";
    int batchsize = 1;
    while ((opt = getopt_long_only(argc,argv,"i:o:m:c:b:",opts,&option_index))!=-1){
        switch (opt)
        {
        case 'i':onnx=string(optarg);
            break;
        case 'o':engine=string(optarg);
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
        case 'c':califile=string(optarg);
            break;
        case 'b':batchsize=atoi(optarg);
            break;
        default:
            break;
        }
    }

    cout<<"input-onnx "<<onnx<<endl
            <<"output-engine "<<engine<<endl
            <<"batchsize"<<batchsize<<endl;

    if(batchsize<=0){
        cerr<<"batch size can not less than zero"<<endl;
        exit(1);
    }
    
    Tn::onnx2tensorrt net(onnx,1,califile,mode,batchsize);
    net.saveEngine(engine);
    cout<<"save "<<engine<<endl;

    return 0;

}