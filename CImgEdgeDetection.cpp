#include "CImg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <linux/limits.h>
#include <list>
#include <iostream>

using namespace cimg_library;
using namespace std;

const char* PATH = "./img";
const char* FILENAME = "caixa.jpg";

const CImg<char> SOBEL_KERNEL_X(3,3,1,1,{
		-1,0,1,
		-2,0,2,
		-1,0,1},false);

const CImg<char> SOBEL_KERNEL_Y(3,3,1,1,{
		 1 , 2 , 1,
		 0 , 0 , 0,
		-1 ,-2 ,-1},false);

const CImg<double> GAUSSIAN_KERNEL_AUX(5,5,1,1,{
		2 ,  4 ,  5 ,  4 , 2,
		4 ,  9 , 12 ,  9 , 4,
		5 , 12 , 15 , 12 , 5,
		4 ,  9 , 12 ,  9 , 4,
		2 ,  4 ,  5 ,  4 , 2}, false);

const CImg<double> GAUSSIAN_KERNEL = GAUSSIAN_KERNEL_AUX / 159;

const double EDGE = 255;
const double SUPPRESS = 0;

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

inline char* getFileName(const char* path, const char* filename)
{
	if (!path || !filename)
		return NULL;

	char fullpath[PATH_MAX];

	realpath(path, fullpath);

	char* fullFileName = (char*) calloc(strlen(fullpath) + strlen(filename) + 1, sizeof(char));

	if (fullFileName)
		sprintf(fullFileName, "%s/%s", fullpath, filename);

	return fullFileName;
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
 */

inline void hysteresis(CImg<double> &G, CImg<double> &edgeTrace, int x, int y, double threshold);
inline void checkNeighborhood(list<pair<int, int>> &edges, CImg<double> &G, CImg<double> &edgeTrace, int x, int y, double threshold);

CImg<double> canny(CImg<double>& grayscaleImg)
{
	CImg<double> gaussian  = grayscaleImg.get_convolve(GAUSSIAN_KERNEL);

	CImg<double> Gx = gaussian.get_convolve(SOBEL_KERNEL_X);
	CImg<double> Gy = gaussian.get_convolve(SOBEL_KERNEL_Y);
	CImg<double> G  = ( Gx.get_sqr().normalize(0,255) + Gy.get_sqr().normalize(0,255) ).cut(0, 255).sqrt().normalize(0, 255);
	CImg<double> Atan2 = Gy.get_atan2(Gx);

	// Non-maximum suppression

	const double SECTOR = 22.5;
	const double ANGLE[4] = {0, 45, 90, 135};

	cimg_forXY(Atan2,x,y)
	{
		double angle = toDegrees(Atan2(x, y));

		for (int i = 0; i < 4; i++)
		{
			double lowLimit1  = angleSum(ANGLE[i]    , -SECTOR);
			double lowLimit2  = angleSum(ANGLE[i]+180, -SECTOR);
			double highLimit1 = angleSum(ANGLE[i]    ,  SECTOR);
			double highLimit2 = angleSum(ANGLE[i]+180,  SECTOR);

			if ((angle > lowLimit1 && angle <= highLimit1) ||
                (angle > lowLimit2 && angle <= highLimit2))
			{
				angle = ANGLE[i];
				break;
			}
		}

		Atan2(x, y) = angle;
	}

	const double highThreshold = 140;
	const double lowThreshold  = 25;

	CImg_3x3(I,double);
	cimg_for3x3(G,x,y,0,0,I,double)
	{
		double g = G(x,y);
		double angle = Atan2(x,y);

		if (angle == 0)
			g = (g < Ipc || g < Inc) ? SUPPRESS : g;
		else if (angle == 45)
			g = (g < Inp || g < Ipn) ? SUPPRESS : g;
		else if (angle == 90)
			g = (g < Icp || g < Icn) ? SUPPRESS : g;
		else // angle == 135
			g = (g < Ipp || g < Inn) ? SUPPRESS : g;

		G(x, y) = g;
	}

	// Edge tracking by hysteresis

	CImg<double> edgeTrace = G.get_fill(0);

	cimg_forXY(G,x,y)
	{
		if (edgeTrace(x, y) != EDGE && G(x, y) >= highThreshold)
		{
			edgeTrace(x,y) = EDGE;
			hysteresis(G, edgeTrace, x, y, lowThreshold);
		}
	}

	return edgeTrace;
}

inline void hysteresis(CImg<double> &G, CImg<double> &edgeTrace, int x, int y, double threshold)
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

inline void checkNeighborhood(list<pair<int, int>> &edges, CImg<double> &G, CImg<double> &edgeTrace, int x, int y, double threshold)
{
	if (x > 0 && y > 0 && edgeTrace(x-1, y-1) != EDGE && G(x-1,y-1) >= threshold)
	{
		edgeTrace(x-1, y-1) = EDGE;
		edges.push_back(pair<int,int>(x-1, y-1));
	}

	if (y > 0 && edgeTrace(x, y-1) != EDGE && G(x, y-1) >= threshold)
	{
		edgeTrace(x, y-1) = EDGE;
		edges.push_back(pair<int,int>(x, y-1));
	}

	if (x < G.width()-1 && y > 0 && edgeTrace(x+1, y-1) != EDGE && G(x+1, y-1) >= threshold)
	{
		edgeTrace(x+1, y-1) = EDGE;
		edges.push_back(pair<int,int>(x+1, y-1));
	}

	if (x < G.width()-1 && edgeTrace(x+1, y) != EDGE && G(x+1, y) > threshold)
	{
		edgeTrace(x+1, y) = EDGE;
		edges.push_back(pair<int,int>(x+1, y));
	}

	if (x < G.width()-1 && y < G.height()-1 && edgeTrace(x+1, y+1) != EDGE && G(x+1, y+1) >= threshold)
	{
		edgeTrace(x+1, y+1) = EDGE;
		edges.push_back(pair<int,int>(x+1, y+1));
	}

	if (y < G.height()-1 && edgeTrace(x, y+1) != EDGE && G(x, y+1) >= threshold)
	{
		edgeTrace(x, y+1) = EDGE;
		edges.push_back(pair<int,int>(x, y+1));
	}

	if (x > 0 && y < G.height()-1 && edgeTrace(x-1, y+1) != EDGE && G(x-1, y+1) >= threshold)
	{
		edgeTrace(x-1, y+1) = EDGE;
		edges.push_back(pair<int,int>(x-1, y+1));
	}

	if (x > 0 && edgeTrace(x-1, y) != EDGE && G(x-1, y) >= threshold)
	{
		edgeTrace(x-1, y) = EDGE;
		edges.push_back(pair<int,int>(x-1, y));
	}
}

int main(int argc, char **argv)
{
	CImgList<double> displayList;

	if (char* filename = getFileName(PATH, FILENAME))
	{
		try
		{
			CImg<double> img(filename);
			CImg<double> grayImg = img.get_norm().normalize(0,255);
			displayList.push_back((img,sobel(grayImg),canny(grayImg)));
		}
		catch(exception &ex)
		{
			std::cout << "Sobel error:" << ex.what();
		}
		free(filename);
	}

	try
	{
		CImg<double> img(640,480);
		CImg<double> grayImg = img.load_camera(0,0,false,640,480).get_norm().normalize(0,255);
		displayList.push_back((img,sobel(grayImg),canny(grayImg)));
	}
	catch(exception &ex)
	{
		std::cout << "Canny error:" << ex.what();
	}
	displayList.display();

	return 0;
}
