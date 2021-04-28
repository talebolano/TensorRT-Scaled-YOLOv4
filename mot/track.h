#ifndef _TRACK_H_
#define _TRACK_H_

#include <vector>

namespace track{

    using namespace std;

    enum class TrackState{
        New,
        Tracked,
        Lost,
        Removed
    };

    const float chi2inv95[9] = {
        3.8415,
        5.9915,
        7.8147,
        9.4877,
        11.070,
        12.592,
        14.067,
        15.507,
        16.919        
    };


    class STrack{
        public:
            
            int mtrack_id;
            int mframe_id;
            int mtracklet_len;
            int mstart_frame;
            
            int mclasses;
            float mscores;

            TrackState mstate;
            vector<float> mtlwh;//4
            vector<float> mean;//8
            vector<vector<float>> covariance;//8,8

            STrack(vector<float> tlwh,float score,int classes);

            ~STrack(){};


            void activate(const int frame_id);
            void re_activate(const int frame_id,STrack det);
            void update(const int frame_id,STrack det);
            void predict();
            void mark_lost();
            void mark_removed();
            vector<float> tlwh_to_xyah();
            vector<float> get_tlbr();
        private:
            static int _count;
            static int next_id();

    };

    class SortedTracker
    {
    private:
        int mframed_id;
        vector<STrack> mtracked_stracks;
        vector<STrack> lost_stracks;
        vector<STrack> removed_stracks;
        
    public:
        int max_time_lost;
        SortedTracker(int buffer_size);
        ~SortedTracker();

        vector<vector<float>> update(vector<vector<float>> results);
    };    

}


#endif