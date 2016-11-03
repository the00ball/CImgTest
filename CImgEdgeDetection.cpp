#include "CImg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <linux/limits.h>

using namespace cimg_library;

const char* PATH = "./img";
const char* FILENAME = "caixa.jpg";

const CImg<char> SOBEL_KERNEL_X(3,3,1,1,{-1,0,1,
										 -2,0,2,
										 -1,0,1},false);

const CImg<char> SOBEL_KERNEL_Y(3,3,1,1,{ 1 , 2 , 1,
										  0 , 0 , 0,
										 -1 ,-2 ,-1},false);

const CImg<double> GAUSSIAN_KERNEL_AUX(5,5,1,1,{2 ,  4 ,  5 ,  4 , 2,
												4 ,  9 , 12 ,  9 , 4,
												5 , 12 , 15 , 12 , 5,
												4 ,  9 , 12 ,  9 , 4,
												2 ,  4 ,  5 ,  4 , 2}, false);

const CImg<double> GAUSSIAN_KERNEL = GAUSSIAN_KERNEL_AUX / 159;

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

CImg<double> canny(CImg<double>& grayscaleImg)
{
	CImg<double> gaussian  = grayscaleImg.get_convolve(GAUSSIAN_KERNEL);

	CImg<double> Gx = gaussian.get_convolve(SOBEL_KERNEL_X);
	CImg<double> Gy = gaussian.get_convolve(SOBEL_KERNEL_Y);
	CImg<double> G  = ( Gx.get_sqr().normalize(0,255) + Gy.get_sqr().normalize(0,255) ).cut(0, 255).sqrt().normalize(0, 255);
	CImg<double> Atan2 = Gy.get_atan2(Gx);

	// Non-maximum suppression

	const double SECTION = 22.5;
	const double ANGLE[4] = {0, 45, 90, 135};

	cimg_forXY(Atan2,x,y)
	{
		double angle = toDegrees(Atan2(x, y));

		for (int i = 0; i < 4; i++)
		{
			double lowLimit1  = angleSum(ANGLE[i]    , -SECTION);
			double lowLimit2  = angleSum(ANGLE[i]+180, -SECTION);
			double highLimit1 = angleSum(ANGLE[i]    ,  SECTION);
			double highLimit2 = angleSum(ANGLE[i]+180,  SECTION);

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
			g = (g < Ipc || g < Inc) ? 0 : g;
		else if (angle == 45)
			g = (g < Inp || g < Ipn) ? 0 : g;
		else if (angle == 90)
			g = (g < Icp || g < Icn) ? 0 : g;
		else // angle == 135
			g = (g < Ipp || g < Inn) ? 0 : g;

		// double Threshold

		G(x, y) = g > highThreshold ? 255 : g < lowThreshold ? 0 : g;
	}

	return G;
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
		catch(...)
		{
		}
		free(filename);
	}

	CImg<double> img(640,480);
	CImg<double> grayImg = img.load_camera(0,0,false,640,480).get_norm().normalize(0,255);
	displayList.push_back((img,sobel(grayImg),canny(grayImg)));

	displayList.display();

	return 0;
}
