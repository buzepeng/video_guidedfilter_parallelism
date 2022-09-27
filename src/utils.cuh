#include <opencv2/opencv.hpp>

#define TILE_W      8
#define TILE_H      8
#define RADIUS      6
#define EPS         0.005
#define DIAM        (RADIUS*2+1)
#define SIZE        (DIAM*DIAM)
#define BLOCK_W     (TILE_W+(2*RADIUS))
#define BLOCK_H     (TILE_H+(2*RADIUS))

__device__ float3 operator+(float3 a, float3 b){
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ float3 operator+(float3 a, float b){
    return make_float3(a.x+b, a.y+b, a.z+b);
}
__device__ float3 operator-(float3 a, float3 b){
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ float3 operator*(float3 a, float3 b){
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ float3 operator/(float3 a, float3 b){
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}
__device__ float3 operator/(float3 a, int n){
    return make_float3(a.x/(float)n, a.y/(float)n, a.z/(float)n);
}
__device__ float3 fmaxf3(float3 a, float b){
    return make_float3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}

__device__ void box_filter(float3 *in, float3 *out, int width, int height)
{
    __shared__ float3 smem[BLOCK_W*BLOCK_H];
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;
    x = max(0, x);
    x = min(x, width-1);
    y = max(y, 0);
    y = min(y, height-1);
    const int idx = y * width + x;


    unsigned int bindex = threadIdx.y * blockDim.y + threadIdx.x;

    smem[bindex] = in[idx];
    __syncthreads();

    float3 sum = make_float3(0,0,0);
    int count = 0;
 
    if ((threadIdx.x >= RADIUS) && (threadIdx.x < (BLOCK_W - RADIUS)) &&
            (threadIdx.y >= RADIUS) && (threadIdx.y < (BLOCK_H - RADIUS))) {
       for(int dy = -RADIUS; dy <= RADIUS; dy++) {
            for(int dx = -RADIUS; dx <= RADIUS; dx++) {
                float3 i = smem[bindex + (dy * blockDim.x) + dx];
                sum = sum + i;
                ++count;
            }
        }
        out[idx] = sum / count;
    }
}

__device__ void compute_cov_var(float3 *mean_Ip, float3 *mean_II, float3 *mean_I,
        float3 *mean_p, float3 *var_I, float3 *cov_Ip, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float3 m_I = mean_I[idx];
    var_I[idx] = fmaxf3(mean_II[idx] - m_I * m_I, 0.);
    cov_Ip[idx] = fmaxf3(mean_Ip[idx] - m_I * mean_p[idx], 0.);
}

__device__ void compute_ab(float3 *var_I, float3 *cov_Ip, float3 *mean_I,
        float3 *mean_p, float3 *a, float3 *b, float eps, int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float3 a_ = cov_Ip[idx] / (var_I[idx] + eps);
    a[idx] = a_;
    b[idx] = mean_p[idx] - a_ * mean_I[idx];
}

__device__ void compute_q(float3 *in, float3 *mean_a, float3 *mean_b, float3 *q,
        int width, int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    float3 im_ = in[idx];
    q[idx] = mean_a[idx] * im_ + mean_b[idx];
}

__global__ void mean_kernel(float3* d_input,
        float3 *d_p,
        float3 *d_q,
        float3 *mean_I,
        float3 *mean_p,
        float3 *mean_Ip,
        float3 *mean_II,
        float3 *var_I,
        float3 *cov_Ip,
        float3 *a,
        float3 *b,
        float3 *d_tmp,
        float3 *d_tmp2,
        float3 *mean_a,
        float3 *mean_b,
        int width, int height,
        float eps)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    box_filter(d_input, mean_I, width, height);
    box_filter(d_p, mean_p, width, height);
    __syncthreads();
    box_filter(d_tmp, mean_Ip, width, height);
    box_filter(d_tmp2, mean_II, width, height);
    if (x >= 0 && y >= 0 && x < width && y < height) {
    compute_cov_var(mean_Ip, mean_II, mean_I, mean_p, var_I, cov_Ip, width, height);
    compute_ab(var_I, cov_Ip, mean_I, mean_p, a, b, eps, width, height);
    }
}

__global__ void output_kernel(float3* d_input,
        float3 *d_p,
        float3 *d_q,
        float3 *a,
        float3 *b,
        float3 *mean_a,
        float3 *mean_b,
        int width, int height,
        float eps)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    box_filter(a, mean_a, width, height);
    __syncthreads();
    box_filter(b, mean_b, width, height);
    __syncthreads();

    if (x >= 0 && y >= 0 && x < width && y < height) {
        compute_q(d_p, mean_a, mean_b, d_q, width, height);
    }
}

struct uchar2float{
    __device__
    float3 operator()(uchar3 pixel){
        float3 res;
        res.x = (float)pixel.x/255.0;
        res.y = (float)pixel.y/255.0;
        res.z = (float)pixel.z/255.0;
        return res;
    }
};

struct float2uchar{
    __device__
    uchar3 operator()(float3 pixel){
        uchar3 res;
        res.x = uchar(pixel.x*255.0);
        res.z = uchar(pixel.z*255.0);
        res.y = uchar(pixel.y*255.0);
        return res;
    }
};
