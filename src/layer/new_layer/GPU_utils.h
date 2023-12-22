#ifndef SRC_LAYER_GPU_UTILS_H
#define SRC_LAYER_GPU_UTILS_H

__global__ void do_not_remove_this_kernel()
{
    int tx = threadIdx.x;
    tx = tx + 1;
}

__global__ void prefn_marker_kernel()
{
    int tx = threadIdx.x;
    tx = tx + 1;
}

__host__ void GPU_Utils::insert_post_barrier_kernel()
{

    dim3 GridDim(1, 1, 1);
    dim3 BlockDim(1, 1, 1);
    do_not_remove_this_kernel<<<GridDim, BlockDim>>>();
    cudaDeviceSynchronize();
}

__host__ void GPU_Utils::insert_pre_barrier_kernel()
{

    int *devicePtr;
    int x = 1;

    cudaMalloc((void **)&devicePtr, sizeof(int));
    cudaMemcpy(devicePtr, &x, sizeof(int), cudaMemcpyHostToDevice);

    dim3 GridDim(1, 1, 1);
    dim3 BlockDim(1, 1, 1);
    prefn_marker_kernel<<<GridDim, BlockDim>>>();
    cudaFree(devicePtr);
    cudaDeviceSynchronize();
}

// Use this class to define GPU functions that students don't need access to.
class GPU_Utils
{
public:
    /* For creating a dummy kernel call so that we can distinguish between kernels launched for different layers
     * in the Nsight Compute CLI for measuring per layer Op Times
     */
    void insert_post_barrier_kernel();
    // For inserting a marker visible in Nsys so that we can time total student function time
    void insert_pre_barrier_kernel();
};

#endif