#include "CPU_conv.h"
#include <math.h>
#include <iostream>

void conv_forward_cpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  /*
  Modify this function to implement the forward pass described in Chapter 16.
  The code in 16 is for a single image.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct, not fast (this is the CPU implementation.)

  Function paramters:
  y - output
  x - input
  k - kernel
  B - batch_size (number of images in x)
  M - number of output feature maps
  C - number of input feature maps
  H - input height dimension
  W - input width dimension
  K - kernel height and width (K x K)
  */

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Insert your CPU convolution kernel code here

  for (int b = 0; b < B; b++)
  { // for each image in the batch
    for (int m = 0; m < M; m++)
    { // for each output feature maps
      for (int h = 0; h < H_out; h++)
      { // for each output element
        for (int w = 0; w < W_out; w++)
        {
          y4d(b, m, h, w) = 0;

          for (int c = 0; c < C; c++)
          { // sum over all input feature maps/channels
            for (int p = 0; p < K; p++)
            { // KxK filter
              for (int q = 0; q < K; q++)
              {
                y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
              }
            }
          }
        }
      }
    }
  }

#undef y4d
#undef x4d
#undef k4d
}

void Conv_CPU::init()
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

void Conv_CPU::forward(const Matrix &bottom)
{
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  float *x = (float *)bottom.data();
  float *y = (float *)top.data();
  float *k = (float *)weight.data();
  float *b = (float *)bias.data();

  const int B = n_sample;
  const int M = channel_out;
  const int C = channel_in;
  const int K = height_kernel; // Assuming width_kernel is also K

  std::cout << "Conv-CPU==" << std::endl;
  std::cout << *x << std::endl;
  std::cout << *y << std::endl;
  std::cout << *k << std::endl;
  std::cout << *b << std::endl;
  std::cout << B << std::endl;
  std::cout << M << std::endl;
  std::cout << C << std::endl;
  std::cout << K << std::endl;
  // Start timer
  auto start_time = std::chrono::high_resolution_clock::now();

  conv_forward_cpu(y, x, k, B, M, C, height_in, width_in, K);

  // Stop timer
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration = (end_time - start_time);
  std::cout << "Op Time: " << duration.count() << " ms" << std::endl;
}

void Conv_CPU::backward(const Matrix &bottom, const Matrix &grad_top)
{
}

void Conv_CPU::update(Optimizer &opt)
{
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());
  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> Conv_CPU::get_parameters() const
{
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void Conv_CPU::set_parameters(const std::vector<float> &param)
{
  if (static_cast<int>(param.size()) != weight.size() + bias.size())
    throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Conv_CPU::get_derivatives() const
{
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
