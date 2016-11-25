#ifndef CIMGTEST_LIB_KRABS_H_
#define CIMGTEST_LIB_KRABS_H_

#include "../CImg.h"
#include <climits>
#include <utility>
#include <vector>

const cimg_library::CImg<char> kSobelKernelX(3,3,1,1,{
		-1,0,1,
		-2,0,2,
		-1,0,1},false);

const cimg_library::CImg<char> kSobelKernelY(3,3,1,1,{
		 1 , 2 , 1,
		 0 , 0 , 0,
		-1 ,-2 ,-1},false);

const unsigned char kEdge = 255;
const unsigned char kSupress = 0;

struct KrabsRegion
{
	unsigned int label = 0;
	int x0 = INT_MAX;
	int y0 = INT_MAX;
	int x1 = INT_MIN;
	int y1 = INT_MIN;

	int width(){ return x1-x0 > 0 ? x1-x0 : 0; }
	int height(){ return y1-y0 > 0 ? y1-y0 : 0; }
	int area(){ return width()*height(); }
};

//! Sobel edge detection
/**
 * Source: https://en.wikipedia.org/wiki/Sobel_operator
 */
cimg_library::CImg<double> KrabsSobel(const cimg_library::CImg<double>& gray);

inline double ToDegrees(const double radians);

inline double AngleSum(const double angle, const double value);

cimg_library::CImg<unsigned char> Hysteresis(const cimg_library::CImg<double> &gradient, const double high_threshold, const double low_threshold);

inline void CheckNeighborhood(std::vector<std::pair<int, int>> &neighborhood, const cimg_library::CImg<double> &gradient, cimg_library::CImg<unsigned char> &edge_trace, const int x, const int y, const double threshold);


//! Canny edge detection
/**
 * \param gray Image source to edge detection. It must be a grayscale image
 * \param sigma
 * \param low_threshold
 * \param high_threshold
 *
 * (1) Apply Gaussian filter to smooth the image in order to remove the noise
 * (2) Find the intensity gradients of the image
 * (3) Apply non-maximum suppression to get rid of spurious response to edge detection
 * (4) Apply double threshold to determine potential edges
 * (5) Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
 *
 * Source: https://en.wikipedia.org/wiki/Canny_edge_detector
 */
cimg_library::CImg<unsigned char> KrabsCanny(const cimg_library::CImg<double>& gray, const float sigma, const double low_threshold, const double high_threshold);

inline void Labeling(std::vector<std::pair<int, int>> &neighborhood, const cimg_library::CImg<double> &binary, cimg_library::CImg<unsigned int> &labeled, KrabsRegion &region, const int x, const int y, const unsigned int current_label);

//! Labeling using one component at time approach
/**
 * Source: https://en.wikipedia.org/wiki/Connected-component_labeling#One_component_at_a_time
 */
cimg_library::CImg<unsigned int> KrabsLabeling(const cimg_library::CImg<double> &binary, std::vector<KrabsRegion> &regions, const int min_area);

bool KrabsFindButton(const char* filename, std::vector<KrabsRegion> regions, const char* button_name, KrabsRegion& button_region, const float zoom_factor=1.0f);

#endif // CIMGTEST_LIB_KRABS_H_
