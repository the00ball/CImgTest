#include "CImg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <iostream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <string>

using namespace cimg_library;
using namespace std;

const CImg<char> SOBEL_KERNEL_X(3,3,1,1,{
		-1,0,1,
		-2,0,2,
		-1,0,1},false);

const CImg<char> SOBEL_KERNEL_Y(3,3,1,1,{
		 1 , 2 , 1,
		 0 , 0 , 0,
		-1 ,-2 ,-1},false);

const double EDGE = 255;
const double SUPPRESS = 0;
const double NON_LABELED = -1;

struct LabelRegion
{
	int label = NON_LABELED;
	int x0 = -1;
	int y0 = -1;
	int x1 = -1;
	int y1 = -1;
};

inline double toDegrees(const double radians)
{
	return  (radians > 0 ? radians : radians + 2*M_PI) * 180/M_PI;
}

inline double angleSum(double angle, const double value)
{
	angle = angle + value;
	angle = fmod(angle, 360);
	if (angle < 0)
		angle += 360;
	return angle;
}

/**
 * Sobel edge detection
 *
 * Source: https://en.wikipedia.org/wiki/Sobel_operator
 *
 */
CImg<double> sobel(CImg<double>& grayscaleImg)
{
	CImg<double> Gx = grayscaleImg.get_convolve(SOBEL_KERNEL_X).sqr().normalize(0, 255);
	CImg<double> Gy = grayscaleImg.get_convolve(SOBEL_KERNEL_Y).sqr().normalize(0, 255);
	CImg<double> G  = ( Gx + Gy ).cut(0, 255).sqrt().normalize(0, 255);

	return G;
}

/**
 * Canny edge detection
 *
 * Source: https://en.wikipedia.org/wiki/Canny_edge_detector
 *
 * (1) Apply Gaussian filter to smooth the image in order to remove the noise
 * (2) Find the intensity gradients of the image
 * (3) Apply non-maximum suppression to get rid of spurious response to edge detection
 * (4) Apply double threshold to determine potential edges
 * (5) Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
 *
 */

inline void hysteresis(CImg<double> &G, CImg<double> &edgeTrace, const int x, const int y, double threshold);
inline void checkNeighborhood(list<pair<int, int>> &edges, CImg<double> &G, CImg<double> &edgeTrace, const int x, const int y, double threshold);

CImg<double> canny(CImg<double>& grayscaleImg, const float sigma, const double lowThreshold, const double highThreshold)
{
	// (1) Apply Gaussian filter to smooth the image in order to remove the noise

	CImg<double> gaussian = grayscaleImg.get_blur(sigma, true, true);

	// (2) Find the intensity gradients of the image

	CImg<double> Gx = gaussian.get_convolve(SOBEL_KERNEL_X);
	CImg<double> Gy = gaussian.get_convolve(SOBEL_KERNEL_Y);
	CImg<double> G  = ( Gx.get_sqr().normalize(0,255) + Gy.get_sqr().normalize(0,255) ).cut(0, 255).sqrt().normalize(0, 255);
	CImg<double> Atan2 = Gy.get_atan2(Gx);

	// (3) Apply non-maximum suppression to get rid of spurious response to edge detection

	const double SECTOR = 22.5;
	const double ANGLE[4] = {0, 45, 90, 135};

	CImg_3x3(I,double);
	cimg_for3x3(G,x,y,0,0,I,double)
	{
		double g = G(x,y);

		double angle = toDegrees(Atan2(x, y));

		for (int i = 0; i < 4; i++)
		{
			const double lowLimit1  = angleSum(ANGLE[i]    , -SECTOR);
			const double lowLimit2  = angleSum(ANGLE[i]+180, -SECTOR);
			const double highLimit1 = angleSum(ANGLE[i]    ,  SECTOR);
			const double highLimit2 = angleSum(ANGLE[i]+180,  SECTOR);

			if ((!i && (angle > lowLimit1 || angle <= highLimit1)) ||
				(angle > lowLimit1 && angle <= highLimit1) ||
				(angle > lowLimit2 && angle <= highLimit2))
			{
				angle = ANGLE[i];
				break;
			}
		}

		switch(static_cast<int>(angle))
		{
			case   0: g = (g < Ipc || g < Inc) ? SUPPRESS : g; break;
			case  45: g = (g < Inp || g < Ipn) ? SUPPRESS : g; break;
			case  90: g = (g < Icp || g < Icn) ? SUPPRESS : g; break;
			case 135: g = (g < Ipp || g < Inn) ? SUPPRESS : g; break;
		}

		G(x, y) = g;
	}

	// (4) Apply double threshold to determine potential edges
	// (5) Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

	CImg<double> edgeTrace = G.get_fill(0);

	cimg_forXY(G,x,y)
	{
		if (!edgeTrace(x, y) && G(x, y) >= highThreshold)
		{
			edgeTrace(x,y) = EDGE;
			hysteresis(G, edgeTrace, x, y, lowThreshold);
		}
	}

	return edgeTrace;
}

inline void hysteresis(CImg<double> &G, CImg<double> &edgeTrace, const int x, const int y, double threshold)
{
	list<pair<int, int>> edges;

	checkNeighborhood(edges, G, edgeTrace, x, y, threshold);

	while(!edges.empty())
	{
		pair<int,int> point = edges.back();
		edges.pop_back();
		checkNeighborhood(edges, G, edgeTrace, point.first, point.second, threshold);
	}
}

inline void checkNeighborhood(list<pair<int, int>> &edges, CImg<double> &G, CImg<double> &edgeTrace, const int x, const int y, double threshold)
{
	// 8-connected pixels

	if (x > 0 && y > 0 && !edgeTrace(x-1, y-1) && G(x-1,y-1) >= threshold)
	{
		edgeTrace(x-1, y-1) = EDGE;
		edges.push_back(pair<int,int>(x-1, y-1));
	}

	if (y > 0 && !edgeTrace(x, y-1) && G(x, y-1) >= threshold)
	{
		edgeTrace(x, y-1) = EDGE;
		edges.push_back(pair<int,int>(x, y-1));
	}

	if (x < G.width()-1 && y > 0 && !edgeTrace(x+1, y-1) && G(x+1, y-1) >= threshold)
	{
		edgeTrace(x+1, y-1) = EDGE;
		edges.push_back(pair<int,int>(x+1, y-1));
	}

	if (x < G.width()-1 && !edgeTrace(x+1, y) && G(x+1, y) > threshold)
	{
		edgeTrace(x+1, y) = EDGE;
		edges.push_back(pair<int,int>(x+1, y));
	}

	if (x < G.width()-1 && y < G.height()-1 && !edgeTrace(x+1, y+1) && G(x+1, y+1) >= threshold)
	{
		edgeTrace(x+1, y+1) = EDGE;
		edges.push_back(pair<int,int>(x+1, y+1));
	}

	if (y < G.height()-1 && !edgeTrace(x, y+1) && G(x, y+1) >= threshold)
	{
		edgeTrace(x, y+1) = EDGE;
		edges.push_back(pair<int,int>(x, y+1));
	}

	if (x > 0 && y < G.height()-1 && !edgeTrace(x-1, y+1) && G(x-1, y+1) >= threshold)
	{
		edgeTrace(x-1, y+1) = EDGE;
		edges.push_back(pair<int,int>(x-1, y+1));
	}

	if (x > 0 && !edgeTrace(x-1, y) && G(x-1, y) >= threshold)
	{
		edgeTrace(x-1, y) = EDGE;
		edges.push_back(pair<int,int>(x-1, y));
	}
}

/*
 * Labeling using one component at time approach
 *
 * Using recursive method
 */

void innerLabeling(CImg<double> &binaryImg, CImg<double> &labelImg, LabelRegion &region, const int x, const int y, const int currentLabel);

CImg<double> labeling(CImg<double> &binaryImg, list<LabelRegion> &regions)
{
	CImg<double> labelImg = binaryImg.get_fill(NON_LABELED);
	int currentLabel = 0;

	cimg_forXY(binaryImg,x,y)
	{
		if (labelImg(x, y) == NON_LABELED && binaryImg(x, y) == EDGE)
		{
			LabelRegion region;
			region.x0 = region.x1 = x;
			region.y0 = region.y1 = y;
			region.label = ++currentLabel;

			innerLabeling(binaryImg, labelImg, region, x, y, currentLabel);

			if (region.x1-region.x0 > 0 && region.y1-region.y0 > 0)
				regions.push_back(region);
		}
	}

	return labelImg;
}

void innerLabeling(CImg<double> &binaryImg, CImg<double> &labelImg, LabelRegion &region, const int x, const int y, const int currentLabel)
{
	labelImg(x,y) = currentLabel;

	if (x < region.x0)
		region.x0 = x;
	else if (x > region.x1)
		region.x1 = x;

	if (y < region.y0)
		region.y0 = y;
	else if (y > region.y1)
		region.y1 = y;

	// 8-connected pixels

	if (x > 0 && y > 0 && labelImg(x-1, y-1) == NON_LABELED && binaryImg(x-1,y-1) == EDGE)
		innerLabeling(binaryImg, labelImg, region, x-1, y-1, currentLabel);

	if (y > 0 && labelImg(x, y-1) == NON_LABELED && binaryImg(x, y-1) == EDGE)
		innerLabeling(binaryImg, labelImg, region, x, y-1, currentLabel);

	if (x < binaryImg.width()-1 && y > 0 && labelImg(x+1, y-1) == NON_LABELED && binaryImg(x+1, y-1) == EDGE)
		innerLabeling(binaryImg, labelImg, region, x+1, y-1, currentLabel);

	if (x < binaryImg.width()-1 && labelImg(x+1, y) == NON_LABELED && binaryImg(x+1, y) == EDGE)
		innerLabeling(binaryImg, labelImg, region, x+1, y, currentLabel);

	if (x < binaryImg.width()-1 && y < binaryImg.height()-1 && labelImg(x+1, y+1) == NON_LABELED && binaryImg(x+1, y+1) == EDGE)
		innerLabeling(binaryImg, labelImg, region, x+1, y+1, currentLabel);

	if (y < binaryImg.height()-1 && labelImg(x, y+1) == NON_LABELED && binaryImg(x, y+1) == EDGE)
		innerLabeling(binaryImg, labelImg, region, x, y+1, currentLabel);

	if (x > 0 && y < binaryImg.height()-1 && labelImg(x-1, y+1) == NON_LABELED && binaryImg(x-1, y+1) == EDGE)
		innerLabeling(binaryImg, labelImg, region, x-1, y+1, currentLabel);

	if (x > 0 && labelImg(x-1, y) == NON_LABELED && binaryImg(x-1, y) == EDGE)
		innerLabeling(binaryImg, labelImg, region, x-1, y, currentLabel);
}

bool findButton(const char* filename, list<LabelRegion> regions, const char* buttonName, LabelRegion& buttonRegion)
{
	bool buttonFound = false;

	tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
	// Initialize tesseract-ocr with English, without specifying tessdata path
	if (api->Init(NULL, "eng"))
		fprintf(stderr, "Could not initialize tesseract.\n");
	else
	{
		// Open input image with leptonica library
		Pix *image = pixRead(filename);
		api->SetImage(image);

		while(!regions.empty())
		{
			LabelRegion region = regions.back();
			regions.pop_back();

			api->SetRectangle(region.x0, region.y0, region.x1-region.x0, region.y1-region.y0);

			// Get OCR result
			string outText(api->GetUTF8Text());

			std::cout<<"OCR output: "<<outText;

			if (outText.length() > 0 && outText.find(buttonName) != string::npos)
			{
				buttonRegion = region;
				buttonFound = true;
				break;
			}
		}

		// Destroy used object and release memory
		api->End();
		pixDestroy(&image);
	}

	return buttonFound;
}

int main(int argc, char **argv)
{
	cimg_usage("Retrieve command line arguments");
	const char*  filename      = cimg_option("-i","/home/sirineo/Imagens/screenshots/Screenshot_2015-01-07-00-38-33.png","Input image file");
	const char   type          = cimg_option("-t",'C',"Algorithm type: (S)obel or (C)anny");
	const double lowThreshold  = cimg_option("-lt",15.0,"Low threshold");
	const double highThreshold = cimg_option("-ht",40.0,"High threshold");
	const float  sigma         = cimg_option("-s",1.4f,"Sigma");
	const char*  buttonLabel   = cimg_option("-b","","Button label");

	try
	{
		CImgList<double> displayList;
		CImg<double> image;
		CImg<double> grayscale;
		CImg<double> edge;
		CImg<double> label;

		if (strlen(filename))
		{
			CImg<double> input(filename);
			image = input;
			if (image.width() > 640)
				image.resize(640, 640*image.height()/image.width());
		}
		else
		{
			image.resize(640,480);
			image.load_camera(0,0,false,640,480);
		}

		grayscale = image.get_norm().normalize(0,255);

		switch(type)
		{
			case 'S': edge = sobel(grayscale); break;
			case 'C':
			default: edge = canny(grayscale, sigma, lowThreshold, highThreshold); break;
		}

		list<LabelRegion> regions;

		label = labeling(edge, regions);

		LabelRegion button;
		bool found = findButton(filename, regions, buttonLabel, button);

		if (found)
		{
			const double color[] = {0,255,0};
			image.draw_rectangle(button.x0,button.y0,button.x1,button.y1,color,0.2f);
		}

		displayList.push_back(image);
		displayList.display();
	}
	catch(exception &ex)
	{
		std::cout << "Error:" << ex.what();
	}

	return 0;
}
