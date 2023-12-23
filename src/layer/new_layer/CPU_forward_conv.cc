#include "CPU_forward_conv.h"

void conv_forward_cpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    Function paramters : y - output x - input k - kernel B - batch_size(number of images in x) M - number of output feature maps C - number of input feature maps H - input height dimension W - input width dimension K - kernel height and width(K x K)

                                                                                                                                                                                                                               const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // #defs to simplify indexing. Feel free to use them, or create your own.
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // CPU convolution kernel code

    for (int b = 0; b < B; b++)
    {
        for (int m = 0; m < M; m++)
        {
            for (int h = 0; h < H_out; h++)
            {
                for (int w = 0; w < W_out; w++)
                {
                    y4d(b, m, h, w) = 0;
                    for (int c = 0; c < C; c++)
                    {
                        for (int p = 0; p < K; p++)
                        {
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

// void blurImg(uchar3 *inPixels, int width, int height, float *filter, int filterWidth,
//              uchar3 *outPixels,
//              bool useDevice = false, dim3 blockSize = dim3(1, 1), int kernelType = 1)
// {
//     for (int i = 0; i < width * height; i += 1)
//     {
//         float r = 0, g = 0, b = 0;

//         for (int j = 0; j < filterWidth * filterWidth; j += 1)
//         {
//             int x = i / width + (j / filterWidth - filterWidth / 2);
//             int y = i % width + (j % filterWidth - filterWidth / 2);

//             if (x < 0)
//             {
//                 x = 0;
//             }
//             else if (x > width - 1)
//             {
//                 x = width - 1;
//             }

//             if (y < 0)
//             {
//                 y = 0;
//             }
//             else if (y > height - 1)
//             {
//                 y = height - 1;
//             }

//             int flaten = x * width + y;

//             r += inPixels[flaten].x * filter[j];
//             g += inPixels[flaten].y * filter[j];
//             b += inPixels[flaten].z * filter[j];
//         }
//         outPixels[i].x = r;
//         outPixels[i].y = g;
//         outPixels[i].z = b;
//     }
// }