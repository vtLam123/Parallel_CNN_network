#include "My_GPU.h"

#include <cuda_runtime.h>

// CUDA kernel for convolutionssss
__global__ void conv_forward_gpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out)
{
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;

    if (b < B && m < M && h < H_out && w < W_out)
    {
        float sum = 0;
        for (int c = 0; c < C; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    int x_index = ((b * C + c) * H + h + p) * W + w + q;
                    int k_index = ((m * C + c) * K + p) * K + q;
                    sum += x[x_index] * k[k_index];
                }
            }
        }
        y[((b * M + m) * H_out + h) * W_out + w] = sum;
    }
}

// Function to call the CUDA kernel
__host__ void MyGPU::conv_forward_gpu_caller(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    dim3 blocks(B, M);
    dim3 threads(H_out, W_out);

    conv_forward_gpu<<<blocks, threads>>>(y, x, k, B, M, C, H, W, K, H_out, W_out);

    cudaDeviceSynchronize();
}
