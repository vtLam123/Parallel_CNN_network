#include "gpu.h"

void GPUInterface::im2col1(const Vector &image, Matrix &data_col)
{
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;
    // im2col
    data_col.resize(hw_out, hw_kernel * channel_in);
    #pragma omp parallel for
    for (int c = 0; c < channel_in; c++)
    {
        Vector map = image.block(hw_in * c, 0, hw_in, 1); // c-th channel map
        for (int i = 0; i < hw_out; i++)
        {
            int step_h = i / width_out;
            int step_w = i % width_out;
            int start_idx = step_h * width_in * stride + step_w * stride; // left-top idx of window
            for (int j = 0; j < hw_kernel; j++)
            {
                int cur_col = start_idx % width_in + j % width_kernel - pad_w; // col after padding
                int cur_row = start_idx / width_in + j / width_kernel - pad_h;
                if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
                    cur_row >= height_in)
                {
                    data_col(i, c * hw_kernel + j) = 0;
                }
                else
                {
                    // int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
                    int pick_idx = cur_row * width_in + cur_col;
                    data_col(i, c * hw_kernel + j) = map(pick_idx); // pick which pixel
                }
            }
        }
    }
}

/* 
// CUDA kernel for im2col
__global__ void im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize,
    const int pad, const int stride, const int height_col,
    const int width_col, float* data_col) {
    // Implement your CUDA kernel here
}

// CUDA kernel for forward
__global__ void forward_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize,
    const int pad, const int stride, const int height_col,
    const int width_col, float* data_col) {
    // Implement your CUDA kernel here
}

// Then you can call these kernels in your functions
void Conv::im2col(const Vector &image, Matrix &data_col) {
    // Call im2col_gpu_kernel
}

void Conv::forward(const Matrix &bottom) {
    // Call forward_gpu_kernel
}

 */