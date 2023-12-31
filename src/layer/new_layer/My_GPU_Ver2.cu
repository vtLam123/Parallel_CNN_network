#include "My_GPU.h"

#include <cuda_runtime.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Declare the constant memory for the filter
__constant__ float const_k[TILE_WIDTH][TILE_WIDTH];

__global__ void conv_forward_gpu(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out)
{
    // Shared memory for input tiles
    __shared__ float ds_x[TILE_WIDTH][TILE_WIDTH];

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
        // Load input tiles into shared memory
        for (int i = threadIdx.y; i < K; i += blockDim.y)
        {
            for (int j = threadIdx.x; j < K; j += blockDim.x)
            {
                ds_x[i][j] = x[(b * C + c) * H * W + (h + i) * W + (w + j)];
            }
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Accumulate the product of the input pixel and the filter coefficient
        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                sum += ds_x[p][q] * const_k[p][q];
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

    // Allocate device memory for input and output
    float *d_x, *d_y;
    cudaMalloc(&d_x, B * C * H * W * sizeof(float));
    cudaMalloc(&d_y, B * M * H_out * W_out * sizeof(float));

    // Copy input from host to device
    cudaMemcpy(d_x, x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);

    // Copy filter from host to constant memory
    cudaMemcpyToSymbol(const_k, k, M * C * K * K * sizeof(float));

    // Define the grid and block dimensions
    dim3 gridDim((W_out - 1) / TILE_WIDTH + 1, (H_out - 1) / TILE_WIDTH + 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, B * M);

    // Launch the kernel
    conv_forward_gpu<<<gridDim, blockDim>>>(d_y, d_x, B, M, C, H, W, K, H_out, W_out);

    // Copy output from device to host
    cudaMemcpy(y, d_y, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}