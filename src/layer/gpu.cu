#include <cmath>
#include <iostream>
#include "gpu.h"


// __global__ void im2col_gpu_kernel(const int n, const float* data_im,
//     const int height, const int width, const int kernel_h, const int kernel_w,
//     const int pad_h, const int pad_w, const int stride_h, const int stride_w,
//     const int height_col, const int width_col, float* data_col) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   for (; index < n; index += blockDim.x * gridDim.x) {
//     int w_out = index % width_col;
//     int h_index = index / width_col;
//     int h_out = h_index % height_col;
//     int channel_in = h_index / height_col;
//     int channel_out = channel_in * kernel_h * kernel_w;
//     int h_in = h_out * stride_h - pad_h;
//     int w_in = w_out * stride_w - pad_w;
//     float* data_col_ptr = data_col;
//     data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
//     const float* data_im_ptr = data_im;
//     data_im_ptr += (channel_in * height + h_in) * width + w_in;
//     for (int i = 0; i < kernel_h; ++i) {
//       for (int j = 0; j < kernel_w; ++j) {
//         int h = h_in + i;
//         int w = w_in + j;
//         *data_col_ptr =
//             (h >= 0 && w >= 0 && h < height && w < width) ?
//             data_im_ptr[i * width + j] : 0;
//         data_col_ptr += height_col * width_col;
//       }
//     }
//   }
// }

// __host__ void GPUInterface::im2col_gpu(const float* data_im, const int channels,
//     const int height, const int width, const int kernel_h, const int kernel_w,
//     const int pad_h, const int pad_w, const int stride_h, const int stride_w,
//     float* data_col) {
//   int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
//   int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
//   int num_kernels = channels * height_col * width_col;
//   im2col_gpu_kernel<<<(num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
//       CUDA_NUM_THREADS>>>(
//       num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
//       pad_w, stride_h, stride_w, height_col,
//       width_col, data_col);
// }


__global__ void conv_forward_kernel1(const float* input, const float* weight, float* output, 
    const int num_filters, const int channels, const int height, const int width, 
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, 
    const int stride_h, const int stride_w, const int height_col, const int width_col) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int hw_col = height_col * width_col;
    
    if (index < num_filters * hw_col) {
        int w_col = index % width_col;
        int h_col = (index / width_col) % height_col;
        int c_out = index / hw_col;
        
        float value = 0;
        for (int c_in = 0; c_in < channels; ++c_in) {
            for (int h_kernel = 0; h_kernel < kernel_h; ++h_kernel) {
                for (int w_kernel = 0; w_kernel < kernel_w; ++w_kernel) {
                    int h_in = h_col * stride_h - pad_h + h_kernel;
                    int w_in = w_col * stride_w - pad_w + w_kernel;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int offset = ((c_out * channels + c_in) * kernel_h + h_kernel) * kernel_w + w_kernel;
                        value += weight[offset] * input[(c_in * height + h_in) * width + w_in];
                    }
                }
            }
        }
        output[index] = value;
    }
}

void GPUInterface1::im2col_gpu(const float *data_im, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, float *data_col)
{
}

__host__ void GPUInterface1::conv_forward_gpu1(const float *input, const float *weight, float *output,
                                               const int num_filters, const int channels, const int height, const int width,
                                               const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
                                               const int stride_h, const int stride_w)
{

    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int num_kernels = num_filters * height_col * width_col;
    
    conv_forward_kernel1<<<(num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(
        input, weight, output, num_filters, channels, height, width, kernel_h, kernel_w, 
        pad_h, pad_w, stride_h, stride_w, height_col, width_col);
}
