#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#define TILE_W      16
#define TILE_H      16
#define RADIUS      6
#define EPS         0.005
#define DIAM        (RADIUS*2+1)
#define SIZE        (DIAM*DIAM)
#define BLOCK_W     (TILE_W+(2*RADIUS))
#define BLOCK_H     (TILE_H+(2*RADIUS))
#define PAR_NUM     8

std::mutex mtx;
std::condition_variable produce, consume;
std::queue<cv::Mat> q;
int max_size = 20;
bool finished = false, inited = false;
int width, height, fps, frame_Number;

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

    // float3 sum = make_float3(0, 0, 0);
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

__global__ void mul_kernel(float3 *d_input, 
                    float3 *d_p,
                    float3 *d_tmp, 
                    float3 *d_tmp2,
                    int width,
                    int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    if (x >= 0 && y >= 0 && x < width && y < height) {
        d_tmp[idx] = d_input[idx] * d_p[idx];
        d_tmp2[idx] = d_input[idx] * d_input[idx];
    }
}

__global__ void convert2float_kernel(uchar3 *d_input,
                    float3 *d_output,
                    int width,
                    int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    if (x >= 0 && y >= 0 && x < width && y < height) {
        d_output[idx].x = d_input[idx].x/255.0;
        d_output[idx].y = d_input[idx].y/255.0;
        d_output[idx].z = d_input[idx].z/255.0;
    }
}

__global__ void convert2uchar_kernel(float3 *d_input,
                    uchar3 *d_output,
                    int width,
                    int height)
{
    int x = blockIdx.x * TILE_W + threadIdx.x - RADIUS;
    int y = blockIdx.y * TILE_H + threadIdx.y - RADIUS;

    int idx = y * width + x; 
    if (x >= 0 && y >= 0 && x < width && y < height) {
        d_output[idx].x = (uchar)(d_input[idx].x*255.0);
        d_output[idx].y = (uchar)(d_input[idx].y*255.0);
        d_output[idx].z = (uchar)(d_input[idx].z*255.0);
    }
}

class GuidedFilter{
    float3 *d_input, *d_p, *d_output, *d_mean_I, *d_mean_p, *d_mean_Ip,
           *d_mean_II, *d_var_I, *d_cov_Ip, *d_a, *d_b, *d_mean_a,
           *d_mean_b, *d_tmp, *d_tmp2;
    uchar3 *d_image_input, *d_image_p, *d_image_output;
    int height, width, m, n;
    public:
    GuidedFilter(int w, int h){
        width = w;
        height = h;
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
    ~GuidedFilter(){
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
    void filter(uchar3* image_input, uchar3* image_output, uchar3* image_p, cudaStream_t stream)
    {
        int GRID_W = ceil(width /(float)TILE_W)+1;
        int GRID_H = ceil(height / (float)TILE_H)+1;

        const dim3 block(BLOCK_W, BLOCK_H);
        const dim3 grid(GRID_W, GRID_H);

        cudaMemcpyAsync(d_image_input, image_input, width * height * sizeof(uchar3), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_image_p, image_p, width * height * sizeof(uchar3), cudaMemcpyHostToDevice, stream);
        convert2float_kernel<<<grid, block, 0, stream>>>(d_image_input, d_input, width, height);
        convert2float_kernel<<<grid, block, 0, stream>>>(d_image_p, d_p, width, height);

        mul_kernel<<<grid, block, 0, stream>>>(d_input, d_p, d_tmp, d_tmp2, width, height);

        mean_kernel<<<grid, block, 0, stream>>>(d_input, d_p, d_output, d_mean_I, d_mean_p, d_mean_Ip,d_mean_II, d_var_I, d_cov_Ip, d_a, d_b, d_tmp, d_tmp2, d_mean_a,
        d_mean_b, width, height, EPS);

        output_kernel<<<grid, block, 0, stream>>>(d_input, d_p, d_output, d_a, d_b,d_mean_a, d_mean_b, width, height, EPS);
        convert2uchar_kernel<<<grid, block, 0, stream>>>(d_output, d_image_output, width, height);
        cudaMemcpyAsync(image_output, d_image_output, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost, stream);
    }
};

void consumer(std::string output_file){
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    cv::VideoWriter writer;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator (cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    while(!inited);
    writer.open(output_file.c_str(), codec, fps, cv::Size(width, height), true);
    auto guidedfilter = GuidedFilter(width, height);
    cv::Mat input(height, width, CV_8UC3), output(height, width, CV_8UC3);

    while(!finished){
        std::unique_lock<std::mutex> lck(mtx);
        while(q.size()==0)  consume.wait(lck);
        input = q.front();
        q.pop();
        guidedfilter.filter(input.ptr<uchar3>(), output.ptr<uchar3>(), input.ptr<uchar3>(), stream);
        writer.write(output);
        
        produce.notify_all();
        lck.unlock();
    }
    writer.release();
}

void producer(std::string input_file){
    cv::VideoCapture capture(input_file.c_str());
    if(!capture.isOpened()){
        std::cout<<"could not open the video!\n";
    }
    fps = capture.get(cv::CAP_PROP_FPS);
    height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    frame_Number = capture.get(cv::CAP_PROP_FRAME_COUNT);
    inited = true;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator (cv::cuda::HostMem::AllocType::PAGE_LOCKED));
    cv::Mat frame(height, width, CV_8UC3);
    while(capture.read(frame)){
        std::unique_lock<std::mutex> lck(mtx);
        while(q.size()==max_size)   produce.wait(lck);
        q.emplace(frame);
        consume.notify_all();
        lck.unlock();
    }
    finished = true;

    capture.release();
}

int main(int argc, char *argv[]) {
    std::string input_file = "../data/bigbang.mp4";
    std::string output_file = "../data/bigbang_guided.mp4";
    cudaSetDevice(1);

    auto startTime = std::chrono::system_clock::now();

    std::thread consume_th, produce_th;
    produce_th = std::thread(producer, input_file);
    consume_th = std::thread(consumer, output_file);
    produce_th.join();
    consume_th.join();

    auto endTime = std::chrono::system_clock::now();
    int process_time = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "time:" <<process_time<< "ms, mean time:"<< (float)process_time/frame_Number << "ms"<< std::endl;
 
    return 0;
}
