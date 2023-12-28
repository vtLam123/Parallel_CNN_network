#ifndef SRC_LAYER_GPU_H
#define SRC_LAYER_GPU_H
class GPUInterface1
{
public:
    //void get_device_properties1();
    void    im2col_gpu(const float* data_im, const int channels,
                    const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                    float* data_col);
    void conv_forward_gpu1(const float* input, const float* weight, float* output, 
                                              const int num_filters, const int channels, const int height, const int width, 
                                              const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, 
                                              const int stride_h, const int stride_w);
};

#endif
