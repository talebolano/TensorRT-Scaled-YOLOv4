#ifndef _KLMF_H_
#define _KLMF_H_
#include <vector>

namespace klmf{

    using namespace std;
    vector<float> klmf_initiate_mean(const vector<float> measurement);
    vector<vector<float>> klmf_initiate_covariance(const vector<float> measurement);
    void klmf_update(vector<float> &mean,vector<vector<float>> &covariance,const vector<float> measurement);
    void klmf_predict(vector<float> &mean,vector<vector<float>> &covarinace);
    float gating_distance(const vector<float>mean,const vector<vector<float>>covariance,const vector<float>measurements);
}

#endif