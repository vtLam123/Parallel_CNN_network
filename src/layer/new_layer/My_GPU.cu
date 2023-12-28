#include "My_GPU.h"

__global__ void im2col_kernel(const float *image, float *data_col, int height_in, int width_in, int height_kernel, int width_kernel, int height_out, int width_out, int stride, int pad_h, int pad_w, int channel_in)
{
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hw_out * hw_kernel * channel_in)
        return;

    int c = idx / (hw_out * hw_kernel);
    idx %= hw_out * hw_kernel;
    int i = idx / hw_kernel;
    int j = idx % hw_kernel;

    int step_h = i / width_out;
    int step_w = i % width_out;
    int start_idx = step_h * width_in * stride + step_w * stride;

    int cur_col = start_idx % width_in + j % width_kernel - pad_w;
    int cur_row = start_idx / width_in + j / width_kernel - pad_h;

    if (cur_col < 0 || cur_col >= width_in || cur_row < 0 || cur_row >= height_in)
    {
        data_col[idx] = 0;
    }
    else
    {
        int pick_idx = cur_row * width_in + cur_col;
        data_col[idx] = image[c * hw_in + pick_idx];
    }
}
