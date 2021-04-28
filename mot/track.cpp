#include <vector>
#include <map>
#include <iostream>
#include "klmf.h"
#include "lapjv.h"
#include "track.h"

namespace track{

    inline vector<float> tlbr_to_tlwh(vector<float>tlbr){
        vector<float> tlwh(4);
        tlwh[2] = tlbr[2] - tlbr[0];
        tlwh[3] = tlbr[3] - tlbr[1];
        tlwh[0] = tlbr[0];
        tlwh[1] = tlbr[1];

        return tlwh;
    }

    inline float iou_distance(vector<float> a,vector<float> b){

        float xx1 = max(a[0],b[0]);
        float yy1 = max(a[1],b[1]);
        float xx2 = min(a[2],b[2]);
        float yy2 = min(a[3],b[3]);

        float w = max(0.0f,xx2-xx1);
        float h = max(0.0f,yy2-yy1);
        float inter = w*h;

        float area1 = (a[3]-a[1])*(a[2]-a[0]);
        float area2 = (b[3]-b[1])*(b[2]-b[0]);

        float iou = inter /(area1+area2-inter);

        return 1-iou;
    }

    int STrack::_count = 0;
    int STrack::next_id(){
        _count +=1;
        return _count;
    }


    STrack::STrack(vector<float> tlwh,float score,int classes)
        :mtlwh(tlwh),mscores(score),mclasses(classes),is_activated(false),
        mstate(TrackState::New){

    }

    void STrack::mark_lost(){
        mstate = TrackState::Lost;
    }

    void STrack::mark_removed(){
        mstate = TrackState::Removed;
    }

    void STrack::activate(const int frame_id){
        mtrack_id = next_id();
        mtracklet_len = 0;
        mstate = TrackState::Tracked;

        is_activated = true;
        mframe_id = frame_id;
        mstart_frame = frame_id;
        vector<float> xyah = tlwh_to_xyah();
        mean = klmf::klmf_initiate_mean(xyah);
        covariance = klmf::klmf_initiate_covariance(xyah);

    }

    void STrack::re_activate(const int frame_id,STrack det){
        //mtrack_id = next_id();
        mtracklet_len +=1;
        mstate = TrackState::Tracked;
        klmf::klmf_update(mean,covariance,det.tlwh_to_xyah());
        is_activated = true;
        mframe_id = frame_id;
    }

    void STrack::update(const int frame_id,STrack det){
        mframe_id = frame_id;
        mtracklet_len +=1;
        mstate = TrackState::Tracked;
        klmf::klmf_update(mean,covariance,det.tlwh_to_xyah());
        is_activated = true;
        mscores = det.mscores;
    }

    void STrack::predict(){
        if(mstate!=TrackState::Tracked){
            mean[7] = 0;
        }
        klmf::klmf_predict(mean,covariance);
    }

    vector<float> STrack::tlwh_to_xyah(){
        vector<float> xyah;
        xyah.resize(mtlwh.size());

        xyah[0] = mtlwh[0] + mtlwh[2] / 2;
        xyah[1] = mtlwh[1] + mtlwh[3] / 2;
        xyah[3] = mtlwh[3];
        xyah[2] = mtlwh[2] / mtlwh[3];

        return xyah;

    }

    vector<float> STrack::get_tlbr(){
        vector<float> tlbr(4);
        if(mean.size()!=0){
            float w = mean[2] *mean[3];
            float h = mean[3];
            tlbr[0] = mean[0] - w / 2;
            tlbr[1] = mean[1] - h / 2;
            tlbr[2] = w/2 + mean[0];
            tlbr[3] = h/2 + mean[1];
        }
        else
        {
            tlbr[0] = mtlwh[0];
            tlbr[1] = mtlwh[1];
            tlbr[2] = mtlwh[2] + mtlwh[0];
            tlbr[3] = mtlwh[3] + mtlwh[1];
        }
        return tlbr;
    }
    
    SortedTracker::SortedTracker(int buffer_size):mframed_id(0),max_time_lost(buffer_size)
    {
    }
    
    SortedTracker::~SortedTracker()
    {
    }

    vector<STrack> SortedTracker::join_starcks(vector<STrack>stracksa,vector<STrack>stracksb){
        map<int,int> exists;
        vector<STrack> outs;
        for(int i=0;i<stracksa.size();++i){
            exists[stracksa[i].mtrack_id] = 1;
            outs.push_back(stracksa[i]);
        }
        for(int i=0;i<stracksb.size();++i){
            if(exists.find(stracksb[i].mtrack_id)==exists.end()){
                exists[stracksb[i].mtrack_id] = 1;
                outs.push_back(stracksb[i]);
            }
        }

        return outs;
    }

    vector<STrack> SortedTracker::sub_starcks(vector<STrack>stracksa,vector<STrack>stracksb){
        map<int,STrack> exists;
        vector<STrack> outs;
        for(int i=0;i<stracksa.size();++i){
            exists.insert(pair<int,STrack>(stracksa[i].mtrack_id,stracksa[i]));
        }        
        for(int i=0;i<stracksb.size();++i){
            if(exists.find(stracksb[i].mtrack_id)!=exists.end()){
                exists.erase(stracksb[i].mtrack_id);
            }
        }

        map<int,STrack>::iterator iter = exists.begin();
        while (iter!=exists.end())
        {
            outs.push_back(iter->second);
            iter++;
        }
        
        return outs;
    }

    vector<vector<float>> SortedTracker::update(vector<vector<float>> results){
        
        mframed_id +=1;
        //cout<<"now framed id is : "<<mframed_id<<endl;
        int det_len = results.size();
        vector<STrack> dets;
        //dets.resize(det_len);

        for(int i=0;i<det_len;++i){
            int classes = (int)results[i][5];
            float score = results[i][4];
            vector<float> tlbr(results[i].begin(),results[i].begin()+4);
            
            dets.push_back(STrack(tlbr_to_tlwh(tlbr),score,classes));
        }

        vector<STrack> strack_pool = join_starcks(mtracked_stracks,lost_stracks); //深拷贝
        mtracked_stracks.clear();
        lost_stracks.clear();

        //   predict and get cost
        float gating_threshold = chi2inv95[3];
        vector<vector<float>> cost_matrix(strack_pool.size(),vector<float>(dets.size()));
        
        for(int i=0;i<strack_pool.size();++i){
            strack_pool[i].predict();
            for(int j=0;j<dets.size();++j){
                if(strack_pool[i].mclasses == dets[j].mclasses){
                    float dists = klmf::gating_distance(strack_pool[i].mean,strack_pool[i].covariance,dets[j].tlwh_to_xyah());
                    if(dists>gating_threshold){
                        cost_matrix[i][j] = 1000.f;
                    }
                    else{
                        cost_matrix[i][j] = dists;
                    }
                    
                }
                else{
                    cost_matrix[i][j] = 1000.f;
                }
                
            }
        }
        //linear assignment

        map<int,int> matches;
        vector<int> u_track;
        vector<int> u_detection;
        if(strack_pool.size()>0 and dets.size()>0){
            vector<int> x(strack_pool.size());
            vector<int> y(dets.size());
            lapjy__(cost_matrix,strack_pool.size(),dets.size(),0.7,true,x,y);

            for(int i=0;i<x.size();++i){
                if(x[i]>=0){
                    //matches[i] = x[i];
                    matches.insert(pair<int,int>(i,x[i]));
                }
                else{
                    u_track.push_back(i);
                }
            }
            for(int i=0;i<y.size();++i){
                if(y[i]<0){
                    u_detection.push_back(i);
                }
            }
        }
        else{
            for(int i=0;i<strack_pool.size();++i){
                u_track.push_back(i);
            }
            for(int i=0;i<dets.size();++i){
                u_detection.push_back(i);
            }            
        }
        

        map<int,int>::iterator matches_iter = matches.begin();

        while(matches_iter!=matches.end()){
            if(strack_pool[matches_iter->first].mstate == TrackState::Tracked){
                strack_pool[matches_iter->first].update(mframed_id,dets[matches_iter->second]);//TrackState::activate
                //now_activated_tracks.push_back(strack_pool[i]);
            }
            else{
                strack_pool[matches_iter->first].re_activate(mframed_id,dets[matches_iter->second]);//TrackState::Tracked
                //now_activated_tracks.push_back(strack_pool[i]);
            }
            matches_iter++;
        }

        // 再次关联
        if(u_track.size()!=0 and u_detection.size()!=0){
            vector<vector<float>> cost_matrix_2(u_track.size(),vector<float>(u_detection.size()));
            for(int i=0;i<u_track.size();++i){
                for(int j=0;j<u_detection.size();++j){

                    cost_matrix_2[i][j] = iou_distance(strack_pool[u_track[i]].get_tlbr(),dets[u_detection[j]].get_tlbr());
                }
            }
            //cout<<"after cost matrix"<<endl;
            map<int,int> matches_iou;
            vector<int> u_track_iou;
            vector<int> u_detection_iou;
            vector<int> x_iou(u_track.size());
            vector<int> y_iou(u_detection.size());
            lapjy__(cost_matrix_2,u_track.size(),u_detection.size(),0.5,true,x_iou,y_iou);
            for(int i=0;i<x_iou.size();++i){
                if(x_iou[i]>=0){
                    matches_iou.insert(pair<int,int>(i,x_iou[i]));
                }
                else{
                    u_track_iou.push_back(i);
                }
            }
            for(int i=0;i<y_iou.size();++i){
                if(y_iou[i]<0){
                    u_detection_iou.push_back(i);
                }
            }
            map<int,int>::iterator matches_iouiter = matches_iou.begin();

            //cout<<"after match"<<endl;

            while(matches_iouiter!=matches_iou.end()){
                if(strack_pool[u_track[matches_iouiter->first]].mstate == TrackState::Tracked){
                    strack_pool[u_track[matches_iouiter->first]].update(mframed_id,dets[u_detection[matches_iouiter->second]]);//TrackState::activate
                    //now_activated_tracks.push_back(strack_pool[i]);
                }
                else{
                    strack_pool[u_track[matches_iouiter->first]].re_activate(mframed_id,dets[u_detection[matches_iouiter->second]]);//TrackState::Tracked
                    //now_activated_tracks.push_back(strack_pool[i]);
                }
                matches_iouiter++;

            }
            //cout<<"after matches_iouiter"<<endl;
            
            vector<int> new_u_tracked;
            vector<int> new_u_detectioned;
            for(int i=0;i<u_track_iou.size();++i){
                new_u_tracked.push_back(u_track[u_track_iou[i]]);
            }
            for(int i=0;i<u_detection_iou.size();++i){
                new_u_detectioned.push_back(u_detection[u_detection_iou[i]]);
            }
            u_track.clear();
            u_detection.clear();
            u_track.swap(new_u_tracked);
            u_detection.swap(new_u_detectioned);



        }
        //cout<<"after second "<<endl;

        for(int i=0;i<u_track.size();++i){
            if(strack_pool[u_track[i]].mstate == TrackState::Tracked){
                strack_pool[u_track[i]].mark_lost();//TrackState::Lost
                //now_lost_stracks.push_back(strack_pool[u_track[i]]);
            }
        }
        //new find add strack_pool
        for(int i=0;i<u_detection.size();++i){
            dets[u_detection[i]].activate(mframed_id);//TrackState::Tracked
            strack_pool.push_back(dets[u_detection[i]]);
            //now_activated_tracks.push_back(dets[u_detection[i]]);
        }

        for(int i=0;i<strack_pool.size();++i){
            if(strack_pool[i].mstate==TrackState::Lost){
                if(mframed_id-strack_pool[i].mframe_id > max_time_lost){
                    //cout<<"maybe bug"<<endl;
                    strack_pool[i].mark_removed();//TrackState::Removed
                    //now_reomve_stracks.push_back(lost_stracks[i]);
                }
            }
        }

        for(int i=0;i<strack_pool.size();++i){
            if(strack_pool[i].mstate==TrackState::Tracked){
                mtracked_stracks.push_back(strack_pool[i]);
            }
            else if(strack_pool[i].mstate==TrackState::Lost)
            {
                lost_stracks.push_back(strack_pool[i]);
            }
            
        }

        vector<vector<float>> outs(mtracked_stracks.size(),vector<float>(7));

        for(int i=0;i<mtracked_stracks.size();++i){
            
            vector<float> tlbr = mtracked_stracks[i].get_tlbr(); // x1 y1 x2 y2

            outs[i][0] = tlbr[0];
            outs[i][1] = tlbr[1];
            outs[i][2] = tlbr[2];
            outs[i][3] = tlbr[3];

            outs[i][4] = mtracked_stracks[i].mscores;
            outs[i][5] = (float)mtracked_stracks[i].mclasses;
            outs[i][6] = (float)mtracked_stracks[i].mtrack_id;
        }


        return outs;

    }

}

