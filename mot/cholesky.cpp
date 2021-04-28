#include <vector>
#include <cmath>
#include <algorithm>
#include "cholesky.h"

namespace cholesky{

    using namespace std;

    vector<vector<float>> fuck_cholesky(const vector<vector<float>> matrix){
    /*
    input n*n 正定矩阵
    output n*n 下三角矩阵
    
    */
        const int n = matrix.size();
        vector<vector<float>> lower_diag(n,vector<float>(n));
        
        for(int j=0;j<n;++j){

            float d = 0;

            for(int k=0;k<j;++k){
                float s = 0;
                for(int i=0;i<k;++i){
                    s += lower_diag[k][i] * lower_diag[j][i];
                }
                lower_diag[j][k] = s = (matrix[j][k] - s) / lower_diag[k][k];
                d = d + s*s;

            }
            d = matrix[j][j] - d;

            lower_diag[j][j] = sqrt(d>0?d:0);
            for(int k = j+1;k<n;++k){
                lower_diag[j][k] = 0;
            }


        }

        return lower_diag;

    }


    vector<vector<float>> fuck_cholesky_solve(const vector<vector<float>>lower_diag,const vector<vector<float>> b){
    /*
     solve ax = b

     a(n,n)
     b(n,m)
     x(n,m)

     a = L * L.T

     Ly = b
     L.Tx = y

     output x n,m    
        
    */

        //frist solve Ly = b
        const int n = lower_diag.size();
        const int m = b[0].size();
        vector<vector<float>> y(n,vector<float>(m));
        vector<vector<float>> x(n,vector<float>(m));
        
        for(int i=0;i<m;++i){
            float temp = 0 ;
            for(int j=0;j<n;++j){
                for(int k=0;k<j+1;++k){
                    temp += y[k][i] * lower_diag[j][k];
                }
                if(lower_diag[j][j]!=0){
                    y[j][i] = (b[j][i] -temp) / lower_diag[j][j];
                }
            }


        }


        //secend solve L.T x = y 
        for(int i=0;i<m;++i){
            float temp = 0;
            for(int j=n-1;j>=0;--j){
                
                for(int k=n-1;k>j-1;--k){
                    temp += x[k][i] * lower_diag[k][j];
                }
                if(lower_diag[j][j]!=0){
                    x[j][i] = (y[j][i] - temp) / lower_diag[j][j];
                }

            }

        }


        return x;
    }



}