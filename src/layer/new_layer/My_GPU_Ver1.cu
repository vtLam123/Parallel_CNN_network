#include "My_GPU.h"

#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void conv_forward_gpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out)
{
    // Shared memory for input and filter tiles
    __shared__ float ds_x[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_k[TILE_WIDTH][TILE_WIDTH];

    // Calculate the output pixel coordinates
    int h = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int w = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Calculate the input image and output feature map indices
    int b = threadIdx.z / M;
    int m = threadIdx.z % M;

    // Initialize the output pixel value to zero
    float sum = 0.0f;

    // Loop over the filter coefficients
    for (int c = 0; c < C; c++)
    {
        // Load input and filter tiles into shared memory
        for (int i = threadIdx.y; i < K; i += blockDim.y)
        {
            for (int j = threadIdx.x; j < K; j += blockDim.x)
            {
                ds_x[i][j] = x[(b * C + c) * H * W + (h + i) * W + (w + j)];
                ds_k[i][j] = k[(m * C + c) * K * K + i * K + j];
            }
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Accumulate the product of the input pixel and the filter coefficient
        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                sum += ds_x[p][q] * ds_k[p][q];
            }
        }

        // Synchronize to make sure that the preceding computation is done before loading new tiles
        __syncthreads();
    }

    // Store the output pixel value
    if (h < H_out && w < W_out)
    {
        y[(b * M + m) * H_out * W_out + h * W_out + w] = sum;
    }
}

__host__ void MyGPU::conv_forward_gpu_caller(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Calculate the output image dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Allocate device memory for input, output, and filter
    float *d_x, *d_y, *d_k;
    cudaMalloc(&d_x, B * C * H * W * sizeof(float));
    cudaMalloc(&d_y, B * M * H_out * W_out * sizeof(float));
    cudaMalloc(&d_k, M * C * K * K * sizeof(float));

    // Copy input and filter from host to device
    cudaMemcpy(d_x, x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 gridDim((W_out - 1) / TILE_WIDTH + 1, (H_out - 1) / TILE_WIDTH + 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, B * M);

    // Launch the kernel
    conv_forward_gpu<<<gridDim, blockDim>>>(d_y, d_x, d_k, B, M, C, H, W, K, H_out, W_out);

    // Copy output from device to host
    cudaMemcpy(y, d_y, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_k);
}
