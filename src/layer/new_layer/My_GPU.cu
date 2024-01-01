#include "My_GPU.h"

#include <cuda_runtime.h>

#define TILE_WIDTH 16

// __global__ void conv_forward_gpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out)
// {
//     // Get the block and thread indices
//     int bx = blockIdx.x; // block index along x-axis
//     int by = blockIdx.y; // block index along y-axis
//     int tx = threadIdx.x; // thread index within a block along x-axis

//     // Calculate the output pixel coordinates
//     int h = by; // output pixel row
//     int w = bx; // output pixel column

//     // Calculate the input image and output feature map indices
//     int b = tx / M; // input image index
//     int m = tx % M; // output feature map index

//     // Initialize the output pixel value to zero
//     float sum = 0.0f;

//     // Loop over the filter coefficients
//     for (int c = 0; c < C; c++)
//     {
//         for (int p = 0; p < K; p++)
//         {
//             for (int q = 0; q < K; q++)
//             {
//                 // Calculate the input pixel coordinates
//                 int i = h + p; // input pixel row
//                 int j = w + q; // input pixel column

//                 // Get the input pixel value
//                 float x_val = x[(b * C + c) * H * W + i * W + j];

//                 // Get the filter coefficient
//                 float k_val = k[(m * C + c) * K * K + p * K + q];

//                 // Accumulate the product of the input pixel and the filter coefficient
//                 sum += x_val * k_val;
//             }
//         }
//     }

//     // Store the output pixel value
//     y[(b * M + m) * H_out * W_out + h * W_out + w] = sum;
// }

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int H_grid = ceil(1.0*H_out / TILE_WIDTH);
    int W_grid = ceil(1.0*W_out / TILE_WIDTH); 
    
    int b = blockIdx.x;                 // batch number
    int m = blockIdx.y;                 // output feature
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix
    
    float accum = 0.0f;

    if (h < H_out && w < W_out) 
    {
        for(int c=0; c<C; c++)             // sum over all input features
        {
            for(int p=0; p<K; p++)         // KxK filter 
                for(int q=0; q<K; q++)
                    accum += x[(b) * (C * H * W) + (c) * (H * W) + (h+p) * (W) + w+q] * k[(m) * (C * K * K) + (c) * (K * K) + (p) * (K) + q];
        }
        y[(b) * (M * H_out * W_out) + (m) * (H_out * W_out) + (h) * (W_out) + w] = accum;
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
    dim3 gridDim(W_out, H_out); // grid size is H_out * W_out
    dim3 blockDim(B * M, 1); // block size is B * M

    // Launch the kernel
    //conv_forward_gpu<<<gridDim, blockDim>>>(y, x, k, B, M, C, H, W, K, H_out, W_out);
    conv_forward_kernel<<<gridDim, blockDim>>>(y, x, k, B, M, C, H, W, K);

    // Copy output from device to host
    cudaMemcpy(y, d_y, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_k);
}

