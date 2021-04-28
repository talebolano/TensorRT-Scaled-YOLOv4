#include "nms.h"
#include "config.h"
#include <algorithm>

using namespace std;


bool compare(float cls){
    return cls > vis_thresh;
}
void xywh2xyxy(float* bbox){

    float leftx = bbox[0] - bbox[2]/2;
    float lefty = bbox[1] -bbox[3]/2;
    float rightx = bbox[0] + bbox[2]/2;
    float righty = bbox[1] + bbox[3]/2;

    bbox[0] = leftx;
    bbox[1] = lefty;
    bbox[2] = rightx;
    bbox[3] = righty;
}

float computeIou(vector<float>best_bbox,vector<float>bbox){
    float xx1 = max(best_bbox[0] , bbox[0]);
    float yy1 = max(best_bbox[1] , bbox[1]);
    float xx2 = min(best_bbox[2] , bbox[2]);
    float yy2 = min(best_bbox[3] , bbox[3]);

    float area1 = (best_bbox[2] - best_bbox[0]) *(best_bbox[3] - best_bbox[1]);
    float area2 = (bbox[2] - bbox[0]) *(bbox[3] - bbox[1]);
    
    float w = max(0.0f,xx2-xx1);
    float h = max(0.0f,yy2-yy1);
    float inter = w*h;

    float iou = inter / (area1 + area2 - inter);
    return iou;
}


cv::Scalar labelColor(int label){

    srand(100*label);
    
    int r = rand() % 256;
    int g = rand() % 256;
    int b = rand() % 256;
    
    return cv::Scalar(r,g,b);
}

vector<vector<float>> nms(float*conf,float*cls,float*bbox, int ind_size){

    vector<vector<vector<float>>> nms_bbox_conf_cls; // class num 6

    nms_bbox_conf_cls.reserve(classes);

    for(int c=0;c<classes;++c){
        vector<vector<float>> temp2;
        temp2.reserve(ind_size);
        for(int i=0;i<ind_size;++i){

            cls[i*classes+c] = conf[i] * cls[i*classes+c];
            
            if(cls[i*classes+c]>vis_thresh){
                vector<float> temp(6);
                temp[0] = bbox[i*4] - bbox[i*4+2]/2;
                temp[1] = bbox[i*4+1] -bbox[i*4+3]/2;
                temp[2] = bbox[i*4] + bbox[i*4+2]/2;
                temp[3] = bbox[i*4+1] + bbox[i*4+3]/2;
                temp[4] = cls[i*classes+c];
                
                temp[5] = (float)c;                
                temp2.push_back(temp);
            }
        }
        if(temp2.size()>0){
            nms_bbox_conf_cls.push_back(temp2);
        }

    }
    vector<vector<float>> out_nms; // class num 6
    //nms
    for(int c=0;c<nms_bbox_conf_cls.size();++c){  //对于每一类

        sort(nms_bbox_conf_cls[c].begin(),nms_bbox_conf_cls[c].end(), 
                [](vector<float> i1,vector<float> i2){return i1[4]>i2[4];}); //降序排序，根据conf
        do{
            
            auto iter = nms_bbox_conf_cls[c].begin();
            vector<float> best_bbox = nms_bbox_conf_cls[c][0]; //循环设置最好的bbox
            
            out_nms.push_back(best_bbox); //收集输出的
            nms_bbox_conf_cls[c].erase(iter); //弹出最好的

            for(;iter!=nms_bbox_conf_cls[c].end();){
                float iou = computeIou(best_bbox,*iter);
                if(iou>nms_thresh){
                    nms_bbox_conf_cls[c].erase(iter);
                }
                else
                {
                    iter++;
                }
                
            }
        }while (nms_bbox_conf_cls[c].size()!=0);

    }

    return out_nms;
}


void vis(cv::Mat &img,vector<vector<float>>result){

    cv::Size oriImgSize = img.size();
    int oriH = oriImgSize.height;
    int oriW = oriImgSize.width;

    float scaleX = (oriW*1.0f / inputsize[1]);
    float scaleY = (oriH*1.0f / inputsize[0]);
    scaleX = scaleY = scaleX > scaleY ? scaleX : scaleY;

    float shiftX = (inputsize[1] - oriW/scaleX)/2.f;
    float shiftY = (inputsize[0] - oriH/scaleY)/2.f;

    float label_scale = img.rows * 0.0009;
    int box_think = (img.rows+img.cols) * .001 ;
    int base_line ;

    for(auto iter = result.begin();iter!=result.end();++iter){

        int x1 = (int)((*iter)[0] - shiftX) * scaleX;
        int y1 = (int)((*iter)[1] - shiftY) * scaleY;
        int x2 = (int)((*iter)[2] - shiftX) * scaleX;
        int y2 = (int)((*iter)[3] - shiftY) * scaleY;

        cv::Rect bbox;
        bbox.x = max(x1,0);
        bbox.y = max(y1,0);
        bbox.width = min(x2-x1,oriW);
        bbox.height = min(y2-y1,oriH);

        int labelindex = (int)(*iter)[5];
        string showText = class_names[labelindex];
        float conf = (*iter)[4];

        cv::rectangle(img,bbox,labelColor(labelindex),box_think);
        auto size = cv::getTextSize(showText,cv::FONT_HERSHEY_COMPLEX,label_scale,1,&base_line);
        cv::putText(img,showText,cv::Point(x1,y1-size.height),cv::FONT_HERSHEY_COMPLEX, label_scale , labelColor(labelindex), box_think*2/3, 8, 0);
    }

    return;

}


void plottrack(cv::Mat &img,vector<vector<float>>result){

    cv::Size oriImgSize = img.size();
    int oriH = oriImgSize.height;
    int oriW = oriImgSize.width;

    float scaleX = (oriW*1.0f / inputsize[1]);
    float scaleY = (oriH*1.0f / inputsize[0]);
    scaleX = scaleY = scaleX > scaleY ? scaleX : scaleY;

    float shiftX = (inputsize[1] - oriW/scaleX)/2.f;
    float shiftY = (inputsize[0] - oriH/scaleY)/2.f;

    float label_scale = img.rows * 0.0009;
    int box_think = (img.rows+img.cols) * .001 ;
    int base_line ;

    for(auto iter = result.begin();iter!=result.end();++iter){

        int x1 = (int)((*iter)[0] - shiftX) * scaleX;
        int y1 = (int)((*iter)[1] - shiftY) * scaleY;
        int x2 = (int)((*iter)[2] - shiftX) * scaleX;
        int y2 = (int)((*iter)[3] - shiftY) * scaleY;

        cv::Rect bbox;
        bbox.x = x1;
        bbox.y = y1;
        bbox.width = x2-x1;
        bbox.height = y2-y1;

        int labelindex = (int)(*iter)[5];
        string showText = class_names[labelindex];

        float conf = (*iter)[4];
        int track_id =(int)(*iter)[6];

        showText += string(" id: ")+to_string(track_id);
        cv::rectangle(img,bbox,labelColor(labelindex),box_think);
        auto size = cv::getTextSize(showText,cv::FONT_HERSHEY_COMPLEX,label_scale,1,&base_line);
        cv::putText(img,showText,cv::Point(x1,y1-size.height),cv::FONT_HERSHEY_COMPLEX, label_scale , labelColor(labelindex), box_think*2/3, 8, 0);
    }
    return;

}