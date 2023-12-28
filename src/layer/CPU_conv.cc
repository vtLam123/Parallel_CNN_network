#include "CPU_conv.h"
#include <math.h>
#include <iostream>

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

__global__ void im2col_gpu_kernel(const int n, const float *data_im,
                                  const int height, const int width, const int ksize,
                                  const int pad, const int stride, const int height_col, const int width_col,
                                  float *data_col)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (; index < n; index += blockDim.x * gridDim.x)
  {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    float *data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const float *data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i)
    {
      for (int j = 0; j < ksize; ++j)
      {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr =
            (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void Conv_GPU::im2col(const Vector &image, Matrix &data_col)
{
  int height_col = (height_in + 2 * pad_h - height_kernel) / stride_h + 1;
  int width_col = (width_in + 2 * pad_w - width_kernel) / stride_w + 1;
  int num_kernels = channel_in * height_col * width_col;
  im2col_gpu_kernel<<<(num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
                      CUDA_NUM_THREADS>>>(
      num_kernels, image.data(), height_in, width_in, height_kernel, pad_h,
      stride_h, height_col, width_col, data_col.data());
}

void Conv::forward(const Matrix &bottom)
{
  // Start timer
  auto start_time = std::chrono::high_resolution_clock::now();
  std::cout << "Conv-CPU==" << std::endl;

  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  for (int i = 0; i < n_sample; i++)
  {
    // im2col
    Matrix data_col;
    im2col(bottom.col(i), data_col);
    data_cols[i] = data_col;
    // conv by product
    Matrix result = data_col * weight; // result: (hw_out, channel_out)
    result.rowwise() += bias.transpose();
    top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
  }

  // Stop timer
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration = (end_time - start_time);
  std::cout << "Op Time: " << duration.count() << " ms" << std::endl;
}

// void Conv_CPU::forward(const Matrix &bottom)
// {
//   int n_sample = bottom.cols();
//   top.resize(height_out * width_out * channel_out, n_sample);
//   float *x = (float *)bottom.data();
//   float *y = (float *)top.data();
//   float *k = (float *)weight.data();
//   float *b = (float *)bias.data();

//   const int B = n_sample;
//   const int M = channel_out;
//   const int C = channel_in;
//   const int K = height_kernel; // Assuming width_kernel is also K

//   std::cout << "Conv-CPU==" << std::endl;
//   // Start timer
//   auto start_time = std::chrono::high_resolution_clock::now();

//   conv_forward_cpu(y, x, k, B, M, C, height_in, width_in, K);

//   // Stop timer
//   auto end_time = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<float, std::milli> duration = (end_time - start_time);
//   std::cout << "Op Time: " << duration.count() << " ms" << std::endl;
// }

// col2im, used for grad_bottom
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
// image size: Vector (height_in * width_in * channel_in)
void Conv_CPU::col2im(const Matrix &data_col, Vector &image)
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

void Conv_CPU::backward(const Matrix &bottom, const Matrix &grad_top)
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
