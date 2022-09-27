#include <cuda.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "guidedfilter.cuh"
#include "utils.cuh"

GuidedFilter::GuidedFilter(int w, int h){

    m = width * height * sizeof(uchar3);
    n = width * height * sizeof(float3);

    cudaMalloc<uchar3>(&d_image_input, m);
    cudaMalloc<uchar3>(&d_image_p, m);
    cudaMalloc<uchar3>(&d_image_output, m);

    cudaMalloc<float3>(&d_input, n);
    cudaMalloc<float3>(&d_p, n);
    cudaMalloc<float3>(&d_output, n);
    cudaMalloc<float3>(&d_mean_I, n);
    cudaMalloc<float3>(&d_mean_p, n);
    cudaMalloc<float3>(&d_mean_Ip, n);
    cudaMalloc<float3>(&d_mean_II, n);
    cudaMalloc<float3>(&d_var_I, n);
    cudaMalloc<float3>(&d_cov_Ip, n);
    cudaMalloc<float3>(&d_a, n);
    cudaMalloc<float3>(&d_b, n);
    cudaMalloc<float3>(&d_mean_a, n);
    cudaMalloc<float3>(&d_mean_b, n);
    cudaMalloc<float3>(&d_tmp, n);
    cudaMalloc<float3>(&d_tmp2, n);
}

GuidedFilter::~GuidedFilter(){
    cudaFree(d_image_input);
    cudaFree(d_image_output);
    cudaFree(d_image_p);
    cudaFree(d_input);
    cudaFree(d_p);
    cudaFree(d_output);
    cudaFree(d_mean_I);
    cudaFree(d_mean_p);
    cudaFree(d_mean_Ip);
    cudaFree(d_mean_II);
    cudaFree(d_var_I);
    cudaFree(d_cov_Ip);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_mean_a);
    cudaFree(d_mean_b);
    cudaFree(d_tmp);
    cudaFree(d_tmp2);
}

void GuidedFilter::filter(uchar3* image_input, uchar3* image_output, uchar3* image_p, cudaStream_t stream){
    int GRID_W = ceil(width /(float)TILE_W)+1;
    int GRID_H = ceil(height / (float)TILE_H)+1;
    int width = this->width;
    int height = this->height;

    const dim3 block(BLOCK_W, BLOCK_H);
    const dim3 grid(GRID_W, GRID_H);

    cudaMemcpy(d_image_input, image_input, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_p, image_p, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    thrust::transform(thrust::device, d_image_input, d_image_input+width*height, d_input, uchar2float());
    thrust::transform(thrust::device, d_image_p, d_image_p+width*height, d_p, uchar2float());

    thrust::transform(thrust::device, d_input, d_input+width*height, d_p, d_tmp, [=]__device__(float3 pixel1, float3 pixel2){return pixel1*pixel2;});
    thrust::transform(thrust::device, d_input, d_input+width*height, d_tmp2, [=]__device__(float3 pixel){return pixel*pixel;});

    mean_kernel<<<grid, block>>>(d_input, d_p, d_output, d_mean_I, d_mean_p, d_mean_Ip,d_mean_II, d_var_I, d_cov_Ip, d_a, d_b, d_tmp, d_tmp2, d_mean_a,
    d_mean_b, width, height, EPS);
    cudaDeviceSynchronize();

    output_kernel<<<grid, block>>>(d_input, d_p, d_output, d_a, d_b,d_mean_a, d_mean_b, width, height, EPS);
    cudaDeviceSynchronize();

    thrust::transform(thrust::device, d_output, d_output+width*height, d_image_output, float2uchar());
    cudaMemcpy(image_output, d_image_output, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost);
}