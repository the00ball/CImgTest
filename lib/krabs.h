#ifndef CIMGTEST_LIB_KRABS_H_
#define CIMGTEST_LIB_KRABS_H_

#include <CImg.h>
#include <vector>
#include <math.h>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <string>
#include <limits.h>

using namespace cimg_library;
using namespace std;

const CImg<char> kSobelKernelX(3,3,1,1,{
		-1,0,1,
		-2,0,2,
		-1,0,1},false);

const CImg<char> kSobelKernelY(3,3,1,1,{
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
CImg<double> KrabsSobel(const CImg<double>& gray);

inline double ToDegrees(const double radians);

inline double AngleSum(const double angle, const double value);

CImg<unsigned char> Hysteresis(const CImg<double> &gradient, const double high_threshold, const double low_threshold);

inline void CheckNeighborhood(vector<pair<int, int>> &neighborhood, const CImg<double> &gradient, CImg<unsigned char> &edge_trace, const int x, const int y, const double threshold);


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
CImg<unsigned char> KrabsCanny(const CImg<double>& gray, const float sigma, const double low_threshold, const double high_threshold);

inline void Labeling(vector<pair<int, int>> &neighborhood, const CImg<double> &binary, CImg<unsigned int> &labeled, KrabsRegion &region, const int x, const int y, const unsigned int current_label);

//! Labeling using one component at time approach
/**
 * Source: https://en.wikipedia.org/wiki/Connected-component_labeling#One_component_at_a_time
 */
CImg<unsigned int> KrabsLabeling(const CImg<double> &binary, vector<KrabsRegion> &regions, const int min_area);

bool KrabsFindButton(const char* filename, vector<KrabsRegion> regions, const char* button_name, KrabsRegion& button_region, const float zoom_factor=1.0f);

#endif // CIMGTEST_LIB_KRABS_H_
