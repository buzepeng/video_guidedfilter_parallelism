#include <opencv2/opencv.hpp>

class GuidedFilter{
    float3 *d_input, *d_p, *d_output, *d_mean_I, *d_mean_p, *d_mean_Ip,
           *d_mean_II, *d_var_I, *d_cov_Ip, *d_a, *d_b, *d_mean_a,
           *d_mean_b, *d_tmp, *d_tmp2;
    uchar3 *d_image_input, *d_image_p, *d_image_output;
    int height, width, m, n;
    public:
    GuidedFilter(int w, int h);
    ~GuidedFilter();
    void filter(uchar3* image_input, uchar3* image_output, uchar3* image_p, cudaStream_t stream);
};
