#ifndef _CHOLESKY_H_
#define _CHOLESKY_H_
#include <vector>

namespace cholesky{

    using namespace std;

    vector<vector<float>> fuck_cholesky(const vector<vector<float>> matrix);
    vector<vector<float>> fuck_cholesky_solve(const vector<vector<float>>lower_diag,const vector<vector<float>> b);
}

#endif