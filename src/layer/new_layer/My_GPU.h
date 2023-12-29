#ifndef SRC_LAYER_MY_GPU_H
#define SRC_LAYER_MY_GPU_H
#include <cuda_runtime.h>

// Use this class to define GPU functions that students don't need access to.
class GPU_Support
{
public:
    void conv_forward_gpu_caller(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
};


#endif