#include "mish.h"

__device__ float softplus_kernel(float x, const float threshold = 20) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return expf(x);    // too small
    return logf(expf(x) + 1.);
}
__device__ half softplus_kernel(half x, const half threshold) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return hexp(x);    // too small
    return hlog(hexp(x) + half(1.));
}
__device__ half tanh_activate_kernel(half x){return (half(2.)/(half(1.) + hexp(half(-2.)*x)) - half(1.));}
__device__ float tanh_activate_kernel(float x){return (2./(1. + expf(-2.*x)) - 1.);}

template <typename T>
__global__ void mishKernel( int n, const T* input, T* output, const T MISH_THRESHOLD)
{

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        T x_val = input[idx];
        output[idx] = x_val * tanh_activate_kernel( softplus_kernel(x_val, MISH_THRESHOLD) );
    }
}

int computeMish(cudaStream_t stream, int n, const float* input, float* output)
{

    constexpr int blockSize = 1024;
    const int gridSize = (n + blockSize - 1) / blockSize;
    mishKernel<float><<<gridSize, blockSize, 0, stream>>>(n, input, output,20.);
    return 0;
}
int computeMish(cudaStream_t stream, int n, const half* input, half* output)
{
    const int blockSize = 1024;
    const int gridSize = (n + blockSize - 1) / blockSize;
    mishKernel<half><<<gridSize, blockSize, 0, stream>>>(n, input, output,20.);
    return 0;
}