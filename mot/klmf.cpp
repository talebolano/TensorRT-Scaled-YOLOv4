#include <vector>
#include "klmf.h"
#include "cholesky.h"

namespace klmf{

    using namespace std;

    vector<float> klmf_initiate_mean(const vector<float> measurement){

        const int n = 8;
        vector<float> mean(n);

        for(int i=0;i<n;++i){
            if(i<4){
                mean[i]=measurement[i];
            }
        }
        return mean;
    }

    vector<vector<float>> klmf_initiate_covariance(const vector<float> measurement){
        //get x,p
        const int n = 8;
        vector<vector<float>> covariance(n,vector<float>(n));

        const vector<float> std = {
            2.f / 20 * measurement[3],
            2.f / 20  * measurement[3],
            1e-2f,
            2.f / 20  * measurement[3],
            10.f /160 * measurement[3],
            10.f /160 * measurement[3],
            1e-5f,
            10.f /160 * measurement[3]            
        };

        for(int i=0;i<n;++i){
            for(int j=0;j<n;++j){
                if(i==j){
                    covariance[i][j] = std[j] *std[j];
                }
            }
        }
        return covariance;
    }    

    vector<float> klmf_project_mean(const vector<float> mean){

        // u = Hx 4  4,8 8
        const int n = 4;
        vector<float>::const_iterator first = mean.begin();
        vector<float>::const_iterator last = mean.begin()+n;
        vector<float> project_mean(first,last);
        return project_mean;
    }

    vector<vector<float>> klmf_project_covariance(const vector<float> mean,const vector<vector<float>> covariance){
        //Omiga = HpH.T +R    4,4   4,8 8,8 8,4 4,4
        const int n = 4;
        const vector<float> std = {
            1.f/20 * mean[3],
            1.f/20 * mean[3],
            1e-1f,
            1.f/20 * mean[3]
        };

        vector<vector<float>> project_covarinace(n,vector<float>(n));

        for(int i=0;i<n;++i){
            for(int j=0;j<n;++j){
                if(i == j){
                    project_covarinace[i][j] = covariance[i][j] + std[j] * std[j];
                }
                else
                {
                    project_covarinace[i][j] = covariance[i][j];
                }
                
            }
        }

        return project_covarinace;
    }


    void klmf_update(vector<float> &mean,vector<vector<float>> &covariance,const vector<float> measurement){
        // k.T = (HPH.T+R).T e-1(PH.T).T   (HPH.T+R).T = (HPH.T+R)

        vector<float> project_mean = klmf_project_mean(mean);
        vector<vector<float>> project_covarinace = klmf_project_covariance(mean,covariance);

        vector<vector<float>> lower_diag = cholesky::fuck_cholesky(project_covarinace);

        vector<vector<float>> b_T(4,vector<float>(8));

        for(int i=0;i<8;++i){
            for(int j=0;j<4;++j){
                b_T[j][i] = covariance[i][j];
            }
        }

        // 4,8
        vector<vector<float>> kalman_gain_T = cholesky::fuck_cholesky_solve(lower_diag,b_T);    

        vector<float> innvation(4);

        for(int i=0;i<4;++i){
            innvation[i] = measurement[i] - project_mean[i];
            for(int j=0;j<8;++j){
                mean[j] += innvation[i] * kalman_gain_T[i][j];
                
            }
        }

        for(int i=0;i<8;++i){
            for(int j=0;j<8;++j){
                if(i==j and i<4){
                    covariance[i][j] = covariance[i][j] - kalman_gain_T[i][j]*kalman_gain_T[i][j]*project_covarinace[i][j];
                }
                else if(i==j and i>=4){
                    covariance[i][j] = covariance[i][j] - kalman_gain_T[i-4][j]*kalman_gain_T[i-4][j]*project_covarinace[i-4][j-4];
                }
                else if(i-4==j)
                {
                    covariance[i][j] = covariance[i][j] - kalman_gain_T[i-4][j]*kalman_gain_T[j][i]*project_covarinace[i-4][j];
                }
                else if(j-4 ==i)
                {
                    covariance[i][j] = covariance[i][j] - kalman_gain_T[j-4][i]*kalman_gain_T[i][j]*project_covarinace[i][j-4];
                }
                
            }
        }
        return;
    }

    void klmf_predict(vector<float> &mean,vector<vector<float>> &covarinace){

        const int n = 8;
        const vector<float> std = {
            2.f / 20 * mean[3],
            2.f / 20  * mean[3],
            1e-2f,
            2.f / 20  * mean[3],
            10.f /160 * mean[3],
            10.f /160 * mean[3],
            1e-5f,
            10.f /160 * mean[3]            
        };

        for(int i=0;i<4;++i){
            mean[i] = mean[i] + mean[i+4];
        }

        for(int i=0;i<n;++i){
            for(int j=0;j<n;++j){
                if(i==j and i<4){
                    covarinace[i][j] = covarinace[i][j] + covarinace[4+i][j] +covarinace[j][i+4] +  covarinace[j+4][i+4] + std[j] *std[j];
                }
                else if(i==j and i>=4){
                    covarinace[i][j] = covarinace[i][j] + std[j] *std[j];
                }
                else if(i-4==j)
                {
                    covarinace[i][j] = covarinace[i][j] + covarinace[i][j+4];
                }
                else if(j-4==i)
                {
                    covarinace[i][j] = covarinace[i][j] + covarinace[i+4][j];
                }                

            }
        }


        return;


    }

    float gating_distance(const vector<float>mean,const vector<vector<float>>covariance,const vector<float>measurements){

        vector<float> project_mean = klmf_project_mean(mean);
        vector<vector<float>> project_covarinace = klmf_project_covariance(mean,covariance);

        vector<vector<float>> lower_diag = cholesky::fuck_cholesky(project_covarinace);

        vector<float> d(4);

        for(int i=0;i<4;++i){
            d[i] = measurements[i] - project_mean[i];
        }

        //frist solve Lx = d
        const int n = lower_diag.size(); //4
        const int m = d.size(); //4
        vector<float> x(n);

        float temp = 0;
        for(int i=0;i<n;++i){
            //float temp = 0;
            
            for(int j=0;j<i+1;++j){
                temp += x[j] * lower_diag[i][j];
            }
            if(lower_diag[i][i]!=0){
                x[i] = (d[i] - temp) / lower_diag[i][i];
            }
        }

        float out = 0;
        for(int i=0;i<x.size();++i){
            out += x[i] * x[i];
        }
        return out;
    }


}