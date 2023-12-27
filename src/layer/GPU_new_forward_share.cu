#include <cmath>
#include <iostream>
#include "GPU_new_forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    ////ssss
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

     // Shared memory for input and kernel
    __shared__ float ds_x[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_k[TILE_WIDTH][TILE_WIDTH]; 

    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);

    int b = blockIdx.x;                                       // batch number
    int m = blockIdx.y;                                       // output feature
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix

    float accum = 0.0f;

    for (int i = 0; i < C; i++) {
        // Load input and kernel into shared memory
        ds_x[h][w] = x4d(b, i, h, w);
        ds_k[h][w] = k4d(m, i, h, w);

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Perform the computation
        for (int p = 0; p < K; p++)
            for (int q = 0; q < K; q++)
                accum += ds_x[h + p][w + q] * ds_k[p][q];

        // Synchronize to make sure the computation is done before loading next tile
        __syncthreads();
    }
    if (h < H_out && w < W_out)
    {
    //     for (int c = 0; c < C; c++) // sum over all input features
    //     {
    //         for (int p = 0; p < K; p++) // KxK filter
    //             for (int q = 0; q < K; q++)
    //                 accum += x4d(b, c, h + p, w + q) * k4d(m, c, p, q); // 4 dimensions macro resolve thread index
    //     }
        y4d(b, m, h, w) = accum;
    } // endif (h < H_out && w < W_out)

#undef y4d
#undef x4d
#undef k4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int inputSize = B * C * H * W * sizeof(float);          // input features map is C
    int outputSize = B * M * H_out * W_out * sizeof(float); // output feature map is M
    int maskSize = M * C * K * K * sizeof(float);           // C * M filter Maps of size K*K

    cudaMalloc((void **)device_x_ptr, inputSize);
    cudaMalloc((void **)device_y_ptr, outputSize);
    cudaMalloc((void **)device_k_ptr, maskSize);

    // Copy Inout data to device
    cudaMemcpy(*device_x_ptr, host_x, inputSize, cudaMemcpyHostToDevice);
    // Copy Mask data to device
    cudaMemcpy(*device_k_ptr, host_k, maskSize, cudaMemcpyHostToDevice);

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int Z = H_grid * W_grid;

    // Block dimensions = #of threads in the block
    dim3 numThreadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Grid Dimension = #of Blocks: Batch Size * Num_Output_Features *
    dim3 numBlocksInGrid(B, M, Z);

    int x_title_width = TILE_WIDTH - 1 * K;

    size_t shareMemory = sizeof(float) * (x_title_width * x_title_width + K * K);


    // launch the kernel
    conv_forward_kernel<<<numBlocksInGrid, numThreadsPerBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int outputSize = B * M * H_out * W_out * sizeof(float);

    cudaMemcpy(host_y, device_y, outputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
}


__host__ void GPUInterface::conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data,
                                                  const int num_samples, const int output_channel, const int input_channel,
                                                  const int height_in, const int width_in, const int kernel_height)
{
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight;
    cudaMalloc((void **)&device_input, num_samples * input_channel * height_in * width_in * sizeof(float));  // input features map is input_channel
    cudaMalloc((void **)&device_output, num_samples * output_channel * height_out * width_out * sizeof(float));  // output feature map is output_channel
    cudaMalloc((void **)&device_weight, output_channel * input_channel * kernel_height * kernel_height * sizeof(float));  // input_channel * output_channel filter Maps of size kernel_height * kernel_height

    // Copy input and mask data to device
    cudaMemcpy(device_input, input_data, num_samples * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice);

    // Set the kernel dimensions and call the kernel
    int height_grid = ceil(1.0 * height_out / TILE_WIDTH);
    int width_grid = ceil(1.0 * width_out / TILE_WIDTH);
    int Z = height_grid * width_grid;
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(num_samples, output_channel, Z);

    //

    int x_title_width = TILE_WIDTH - 1 * kernel_height;

    size_t shareMemory = sizeof(float) * (x_title_width * x_title_width + kernel_height * kernel_height);

    // Launch the kernel
    conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block, shareMemory>>>(device_output, device_input, device_weight, num_samples, output_channel, input_channel, height_in, width_in, kernel_height);

    // Copy the output back to host
    cudaMemcpy(output_data, device_output, num_samples * output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_weight);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
