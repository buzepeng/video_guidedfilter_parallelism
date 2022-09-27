#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "guidedfilter.cuh"

std::mutex mtx;
std::condition_variable produce, consume;
std::queue<cv::Mat> q;
int max_size = 20;
bool finished = false, inited = false;
int width, height, fps, frame_Number;

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