#include "GPU_conv.h"
#include <math.h>
#include <iostream>

#define TILE_WIDTH 16

void Conv_Custom::init()
{
    height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
    width_out = (1 + (width_in - width_kernel + 2 * pad_w) / stride);
    dim_out = height_out * width_out * channel_out;

    weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    bias.resize(channel_out);
    grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    grad_bias.resize(channel_out);
    set_normal_random(weight.data(), weight.size(), 0, 0.01);
    set_normal_random(bias.data(), bias.size(), 0, 0.01);
    // std::cout << weight.colwise().sum() << std::endl;
    // std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
}

void Conv_Custom::forward(const Matrix &bottom)
{
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    float *input_data = (float *)bottom.data();
    float *output_data = (float *)top.data();
    float *weight_data = (float *)weight.data();

    const int num_samples = n_sample;
    const int input_channel = channel_in;
    const int output_channel = channel_out;
    const int kernel_height = height_kernel; // Assuming width_kernel is also K

    GPUInterface gpuInterface;
    std::cout << "Convolution - GPU:" << std::endl;

    // Launch marker kernel to aid with student function timing
    // gpuInterface.insert_pre_barrier_kernel();

    // Start layer timer
    GpuTimer timer;
    timer.Start();
    gpuInterface.conv_forward_gpu_full(output_data, input_data, weight_data,
                                       num_samples, output_channel, input_channel,
                                       height_in, width_in, kernel_height);

    // Stop layer timer
    timer.Stop();
    float duration_layer = timer.Elapsed();

    // Launch barrier kernel to aid with timing with nsight-compute
    // gpuInterface.insert_post_barrier_kernel();

    std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv_Custom::im2col(const Vector &image, Matrix &data_col)
{
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;
    // im2col
    data_col.resize(hw_out, hw_kernel * channel_in);
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

void Conv_Custom::col2im(const Matrix &data_col, Vector &image)
{
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;
    // col2im
    image.resize(hw_in * channel_in);
    image.setZero();
    for (int c = 0; c < channel_in; c++)
    {
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
                    continue;
                }
                else
                {
                    // int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
                    int pick_idx = cur_row * width_in + cur_col;
                    image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j); // pick which pixel
                }
            }
        }
    }
}

void Conv_Custom::backward(const Matrix &bottom, const Matrix &grad_top)
{
    int n_sample = bottom.cols();
    grad_weight.setZero();
    grad_bias.setZero();
    grad_bottom.resize(height_in * width_in * channel_in, n_sample);
    grad_bottom.setZero();
    for (int i = 0; i < n_sample; i++)
    {
        // im2col of grad_top
        Matrix grad_top_i = grad_top.col(i);
        Matrix grad_top_i_col = Eigen::Map<Matrix>(grad_top_i.data(),
                                                   height_out * width_out, channel_out);
        // d(L)/d(w) = \sum{ d(L)/d(z_i) * d(z_i)/d(w) }
        grad_weight += data_cols[i].transpose() * grad_top_i_col;
        // d(L)/d(b) = \sum{ d(L)/d(z_i) * d(z_i)/d(b) }
        grad_bias += grad_top_i_col.colwise().sum().transpose();
        // d(L)/d(x) = \sum{ d(L)/d(z_i) * d(z_i)/d(x) } = d(L)/d(z)_col * w'
        Matrix grad_bottom_i_col = grad_top_i_col * weight.transpose();
        // col2im of grad_bottom
        Vector grad_bottom_i;
        col2im(grad_bottom_i_col, grad_bottom_i);
        grad_bottom.col(i) = grad_bottom_i;
    }
}