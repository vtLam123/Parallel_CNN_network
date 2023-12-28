#include "conv.h"
#include <math.h>
#include <iostream>
#include "conv_GPU.h"

void Conv_gpu::init()
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
}

// void Conv::im2col(const Vector &image, Matrix &data_col)
// {
//     int hw_in = height_in * width_in;
//     int hw_kernel = height_kernel * width_kernel;
//     int hw_out = height_out * width_out;
//     // im2col
//     data_col.resize(hw_out, hw_kernel * channel_in);
//     for (int c = 0; c < channel_in; c++)
//     {
//         Vector map = image.block(hw_in * c, 0, hw_in, 1); // c-th channel map
//         for (int i = 0; i < hw_out; i++)
//         {
//             int step_h = i / width_out;
//             int step_w = i % width_out;
//             int start_idx = step_h * width_in * stride + step_w * stride; // left-top idx of window
//             for (int j = 0; j < hw_kernel; j++)
//             {
//                 int cur_col = start_idx % width_in + j % width_kernel - pad_w; // col after padding
//                 int cur_row = start_idx / width_in + j / width_kernel - pad_h;
//                 if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
//                     cur_row >= height_in)
//                 {
//                     data_col(i, c * hw_kernel + j) = 0;
//                 }
//                 else
//                 {
//                     int pick_idx = cur_row * width_in + cur_col;
//                     data_col(i, c * hw_kernel + j) = map(pick_idx); // pick which pixel
//                 }
//             }
//         }
//     }
// }

void Conv_gpu::forward(const Matrix &bottom)
{
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    data_cols.resize(n_sample);
    for (int i = 0; i < n_sample; i++)
    {
        Matrix data_col;
        gpuInterface.im2col1(bottom.col(i), data_col);
        data_cols[i] = data_col;
        Matrix result = data_col * weight; // result: (hw_out, channel_out)
        result.rowwise() += bias.transpose();
        top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
    }
}

void Conv_gpu::col2im(const Matrix &data_col, Vector &image)
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
                    int pick_idx = cur_row * width_in + cur_col;
                    image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j); // pick which pixel
                }
            }
        }
    }
}

void Conv_gpu::backward(const Matrix &bottom, const Matrix &grad_top)
{
    int n_sample = bottom.cols();
    grad_weight.setZero();
    grad_bias.setZero();
    grad_bottom.resize(height_in * width_in * channel_in, n_sample);
    grad_bottom.setZero();
    for (int i = 0; i < n_sample; i++)
    {
        Matrix grad_top_i = grad_top.col(i);
        Matrix grad_top_i_col = Eigen::Map<Matrix>(grad_top_i.data(),
                                                   height_out * width_out, channel_out);
        grad_weight += data_cols[i].transpose() * grad_top_i_col;
        grad_bias += grad_top_i_col.colwise().sum().transpose();
        Matrix grad_bottom_i_col = grad_top_i_col * weight.transpose();
        Vector grad_bottom_i;
        col2im(grad_bottom_i_col, grad_bottom_i);
        grad_bottom.col(i) = grad_bottom_i;
    }
}

void Conv_gpu::update(Optimizer &opt)
{
    Vector::AlignedMapType weight_vec(weight.data(), weight.size());
    Vector::AlignedMapType bias_vec(bias.data(), bias.size());
    Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
    Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

    opt.update(weight_vec, grad_weight_vec);
    opt.update(bias_vec, grad_bias_vec);
}

void Conv_gpu::im2col1(const Vector &image, Matrix &data_col)
{
}

std::vector<float> Conv_gpu::get_parameters() const
{
    std::vector<float> res(weight.size() + bias.size());
    std::copy(weight.data(), weight.data() + weight.size(), res.begin());
    std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
    return res;
}

void Conv_gpu::set_parameters(const std::vector<float> &param)
{
    if (static_cast<int>(param.size()) != weight.size() + bias.size())
        throw std::invalid_argument("Parameter size does not match");
    std::copy(param.begin(), param.begin() + weight.size(), weight.data());
    std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Conv_gpu::get_derivatives() const
{
    std::vector<float> res(grad_weight.size() + grad_bias.size());
    std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
    std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
              res.begin() + grad_weight.size());
    return res;
}
