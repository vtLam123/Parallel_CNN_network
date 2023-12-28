#ifndef SRC_LAYER_MY_GPU_H
#define SRC_LAYER_MY_GPU_H

// // Use this class to define GPU functions that students don't need access to.
// class GPU_Support
// {
// public:
//     void im2col_gpu_kernel(const int n, const float *data_im,
//                            const int height, const int width, const int ksize,
//                            const int pad, const int stride, const int height_col, const int width_col,
//                            float *data_col)
// };

void im2col_kernel(const int n, const float *data_im,
                   const int height, const int width, const int ksize,
                   const int pad, const int stride, const int height_col, const int width_col,
                   float *data_col);

#endif