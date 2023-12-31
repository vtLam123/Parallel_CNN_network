#include "My_GPU.h"

#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void conv_forward_gpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out)
{
    __shared__ float ds_x[TILE_WIDTH + K - 1][TILE_WIDTH + K - 1];
    __shared__ float ds_k[K][K];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h = by * TILE_WIDTH + ty;
    int w = bx * TILE_WIDTH + tx;

    int b = tx / M;
    int m = tx % M;

    float sum = 0.0f;

    for (int c = 0; c < C; c++)
    {
        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                if (h + p < H && w + q < W)
                {
                    ds_x[ty + p][tx + q] = x[(b * C + c) * H * W + (h + p) * W + (w + q)];
                }
                else
                {
                    ds_x[ty + p][tx + q] = 0.0f;
                }

                if (ty < K && tx < K)
                {
                    ds_k[ty][tx] = k[(m * C + c) * K * K + ty * K + tx];
                }
            }
        }

        __syncthreads();

        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                sum += ds_x[ty + p][tx + q] * ds_k[p][q];
            }
        }

        __syncthreads();
    }

    if (h < H_out && w < W_out)
    {
        y[(b * M + m) * H_out * W_out + h * W_out + w] = sum;
    }
}

__host__ void MyGPU::conv_forward_gpu_caller(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    float *d_x, *d_y, *d_k;
    cudaMalloc(&d_x, B * C * H * W * sizeof(float));
    cudaMalloc(&d_y, B * M * H_out * W_out * sizeof(float));
    cudaMalloc(&d_k, M * C * K * K * sizeof(float));

    cudaMemcpy(d_x, x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim((W_out - 1) / TILE_WIDTH + 1, (H_out - 1) / TILE_WIDTH + 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    conv_forward_gpu_optimized<<<gridDim, blockDim>>>(d_y, d_x, d_k, B, M, C, H, W, K, H_out, W_out);

    cudaMemcpy(y, d_y, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_k);
}
