#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <omp.h>
#include <iostream>

#include "guidedfilter.h"

static cv::Mat boxfilter(const cv::Mat &I, int r)
{
    cv::Mat result;
    cv::blur(I, result, cv::Size(r, r), cv::Point(-1, -1), cv::BORDER_REPLICATE);
    return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
    if (mat.depth() == depth)
        return mat;

    cv::Mat result;
    mat.convertTo(result, depth);
    return result;
}

class GuidedFilterImpl
{
public:
    virtual ~GuidedFilterImpl() {}

    cv::Mat filter(const cv::Mat &p, int depth);

protected:
    int Idepth;

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class GuidedFilterMono : public GuidedFilterImpl
{
public:
    GuidedFilterMono(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    int r;
    double eps;
    cv::Mat I, mean_I, var_I;
};

class GuidedFilterColor : public GuidedFilterImpl
{
public:
    GuidedFilterColor(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    std::vector<cv::Mat> Ichannels;
    int r;
    double eps;
    cv::Mat mean_I_r, mean_I_g, mean_I_b;
    cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
};


cv::Mat GuidedFilterImpl::filter(const cv::Mat &p, int depth)
{
    cv::Mat p2 = convertTo(p, Idepth);

    cv::Mat result;
    if (p.channels() == 1)
    {
        result = filterSingleChannel(p2);
    }
    else
    {
        std::vector<cv::Mat> pc;
        cv::split(p2, pc);

        for (std::size_t i = 0; i < pc.size(); ++i)
            pc[i] = filterSingleChannel(pc[i]);

        cv::merge(pc, result);
    }

    return convertTo(result, depth == -1 ? p.depth() : depth);
}

GuidedFilterMono::GuidedFilterMono(const cv::Mat &origI, int r, double eps) : r(r), eps(eps)
{
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);

    Idepth = I.depth();

    mean_I = boxfilter(I, r);
    cv::Mat mean_II = boxfilter(I.mul(I), r);
    var_I = mean_II - mean_I.mul(mean_I);
}

cv::Mat GuidedFilterMono::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = boxfilter(p, r);
    cv::Mat mean_Ip = boxfilter(I.mul(p), r);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

    cv::Mat a = cov_Ip / (var_I + eps); // Eqn. (5) in the paper;
    cv::Mat b = mean_p - a.mul(mean_I); // Eqn. (6) in the paper;

    cv::Mat mean_a = boxfilter(a, r);
    cv::Mat mean_b = boxfilter(b, r);

    return mean_a.mul(I) + mean_b;
}

GuidedFilterColor::GuidedFilterColor(const cv::Mat &origI, int r, double eps) : r(r), eps(eps)
{
    cv::Mat I;
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);

    Idepth = I.depth();

    cv::split(I, Ichannels);

    mean_I_r = boxfilter(Ichannels[0], r);
    mean_I_g = boxfilter(Ichannels[1], r);
    mean_I_b = boxfilter(Ichannels[2], r);

    // variance of I in each local patch: the matrix Sigma in Eqn (14).
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb
    cv::Mat var_I_rr = boxfilter(Ichannels[0].mul(Ichannels[0]), r) - mean_I_r.mul(mean_I_r) + eps;
    cv::Mat var_I_rg = boxfilter(Ichannels[0].mul(Ichannels[1]), r) - mean_I_r.mul(mean_I_g);
    cv::Mat var_I_rb = boxfilter(Ichannels[0].mul(Ichannels[2]), r) - mean_I_r.mul(mean_I_b);
    cv::Mat var_I_gg = boxfilter(Ichannels[1].mul(Ichannels[1]), r) - mean_I_g.mul(mean_I_g) + eps;
    cv::Mat var_I_gb = boxfilter(Ichannels[1].mul(Ichannels[2]), r) - mean_I_g.mul(mean_I_b);
    cv::Mat var_I_bb = boxfilter(Ichannels[2].mul(Ichannels[2]), r) - mean_I_b.mul(mean_I_b) + eps;

    // Inverse of Sigma + eps * I
    invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
    invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
    invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
    invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
    invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
    invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);

    cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);

    invrr /= covDet;
    invrg /= covDet;
    invrb /= covDet;
    invgg /= covDet;
    invgb /= covDet;
    invbb /= covDet;
}

cv::Mat GuidedFilterColor::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = boxfilter(p, r);

    cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(p), r);
    cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(p), r);
    cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(p), r);

    // covariance of (I, p) in each local patch.
    cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
    cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
    cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

    cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
    cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
    cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

    cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b); // Eqn. (15) in the paper;

    return (boxfilter(a_r, r).mul(Ichannels[0])
          + boxfilter(a_g, r).mul(Ichannels[1])
          + boxfilter(a_b, r).mul(Ichannels[2])
          + boxfilter(b, r));  // Eqn. (16) in the paper;
}


GuidedFilter::GuidedFilter(const cv::Mat &I, int r, double eps)
{
    CV_Assert(I.channels() == 1 || I.channels() == 3);

    if (I.channels() == 1)
        impl_ = new GuidedFilterMono(I, 2 * r + 1, eps);
    else
        impl_ = new GuidedFilterColor(I, 2 * r + 1, eps);
}

GuidedFilter::~GuidedFilter()
{
    delete impl_;
}

cv::Mat GuidedFilter::filter(const cv::Mat &p, int depth) const
{
    return impl_->filter(p, depth);
}

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth)
{
    return GuidedFilter(I, r, eps).filter(p, depth);
}


int main(){
    std::string input_file="../data/bigbang.mp4";
    std::string output_file="../data/bigbang_guided.mp4";

    cv::VideoCapture capture(input_file.c_str());
    if(!capture.isOpened()){
    std::cout<<"could not open the video!\n";
    }
    int fps = capture.get(cv::CAP_PROP_FPS);
    int rows = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int cols = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_Number = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    cv::VideoWriter writer;
    writer.open(output_file.c_str(), codec, fps, cv::Size(cols, rows), true);
    int r = 6;
    double eps = 0.005;
    auto startTime = std::chrono::system_clock::now();
    //serial method
    // cv::Mat frame;
    // while(capture.read(frame)){
    //     frame.convertTo(frame, CV_32FC3, 1.0/255.0, 0);
    //     cv::Mat output;
    //     output = guidedFilter(frame, frame, r, eps, -1);
    //     output.convertTo(output, CV_8UC3, 255.0, 0);
    //     writer.write(output);
    // }
    //openMP method
    bool continue_frame = true;
    int par_num = 32;
    while(continue_frame){
        std::vector<cv::Mat> frames;
        for(int frame_cnt = 0; frame_cnt<par_num;frame_cnt++){
            cv::Mat frame;
            if(!capture.read(frame)){
                continue_frame = false;
                break;
            }
            else{
                frame.convertTo(frame, CV_32FC3, 1.0/255.0, 0);
                frames.emplace_back(frame.clone());
            }
        }
        std::vector<cv::Mat> outputs(frames.size(), cv::Mat(rows, cols, CV_8UC3));
        cv::Mat output = cv::Mat(rows, cols, CV_32FC3);
        #pragma omp parallel for private(output) shared(frames)
        for(int i = 0;i<frames.size();i++)
        {   
            // std::cout<<"frame:"<<i<<std::endl;
            output = guidedFilter(frames[i], frames[i], r, eps, -1);
            #pragma omp critical
            output.convertTo(outputs[i], CV_8UC3, 255.0, 0);
        }
        for(int i = 0;i<frames.size();i++)  writer.write(outputs[i]);
    }
    auto endTime = std::chrono::system_clock::now();
    int process_time = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "time:" <<process_time<< "ms, mean time:"<< (float)process_time/frame_Number << "ms"<< std::endl;
    capture.release();
    writer.release();

    return 0;
}