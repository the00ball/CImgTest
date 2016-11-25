#include <CImg.h>
#include <iostream>
#include <vector>

#include "lib/krabs.h"

using namespace std;
using namespace cimg_library;

const double kGreen[] = {0,255,0};
const double kRed[] = {255,0,0};
const int kMaxImageWidth = 300;
const int kResolution[] = {640,480}; // width, height

void DrawRect(const KrabsRegion& region, CImg<double>& image)
{
	image.draw_rectangle(region.x0, region.y0, region.x1, region.y1, kGreen,0.2f);
	image.draw_line(region.x0, region.y0, region.x0, region.y1, kRed, 1);
	image.draw_line(region.x0, region.y0, region.x1, region.y0, kRed, 1);
	image.draw_line(region.x1, region.y0, region.x1, region.y1, kRed, 1);
	image.draw_line(region.x0, region.y1, region.x1, region.y1, kRed, 1);
	image.draw_line(region.x0, region.y0, region.x1, region.y1, kRed, 1);
	image.draw_line(region.x0, region.y1, region.x1, region.y0, kRed, 1);
}

void EdgeDetection(const char* filename, const double low_threshold, const double high_threshold, const float sigma)
{
	CImg<double> image;

	if (strlen(filename))
	{
		image = CImg<>(filename);
		if (image.width() > kMaxImageWidth)
			image.resize(kResolution[0], kResolution[0]*image.height()/image.width());
	}
	else
	{
		image.resize(kResolution[0],kResolution[1]);
		image.load_camera(0,1,false,kResolution[0],kResolution[1]);
	}

	CImg<double> gray = image.get_norm().normalize(0,255);
	(image,KrabsSobel(gray),KrabsCanny(gray, sigma, low_threshold, high_threshold)).display();
}

void FindButton(const char* filename, const double low_threshold, const double high_threshold, const float sigma, const char* button_label, const int min_area)
{
	const bool kLoadFromFile = strlen(filename) > 0;
	const char* kCamFileName = "cam.jpg";
	CImg<double> image;
	float zoom_factor = 1.0f;

	if (kLoadFromFile)
	{
		image = CImg<>(filename);
		if (image.width() > kMaxImageWidth)
		{
			zoom_factor = (float)image.width()/kResolution[0];
			image.resize(kResolution[0],kResolution[0]*image.height()/image.width());
		}
	}
	else
	{
		image.resize(kResolution[0],kResolution[1]);
		image.load_camera(0,1,false,kResolution[0],kResolution[1]);
		FILE* file = cimg::fopen(kCamFileName, "wb+");
		image.save_jpeg(file);
		fclose(file);
	}

	vector<KrabsRegion> region_list;
	KrabsLabeling(KrabsCanny(image.get_norm().normalize(0,255), sigma, low_threshold, high_threshold), region_list, min_area);

	KrabsRegion region;
	if (KrabsFindButton((kLoadFromFile?filename:kCamFileName), region_list, button_label, region, zoom_factor))
		DrawRect(region, image);

	image.display();
}

//! Motion Detection
/**
 *  Based: http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
 */
void MotionDetection(const float sigma, const int mim_area, const double high_threshold, const int dilate, const bool show_threshold)
{
	CImg<double> image(kResolution[0],kResolution[1]);
	CImg<double> first_frame(kResolution[0],kResolution[1]);

	vector<KrabsRegion> region_list;
	CImgDisplay display(image, "Motion Detection");

	first_frame.load_camera(0,1,false,kResolution[0],kResolution[1]).norm().normalize(0,255).blur(sigma,true,true);
	display.display(image);

	while(!display.is_closed())
	{
		image.load_camera(0,0,false,kResolution[0],kResolution[1]);
		CImg<double> gray = image.get_norm().normalize(0,255).blur(sigma,true,true);
		CImg<double> threshold = (first_frame-gray).abs().threshold(high_threshold).dilate(dilate);

		if (show_threshold)
			display = threshold;
		else
		{
			KrabsLabeling(threshold, region_list, mim_area);
			while(!region_list.empty())
			{
				KrabsRegion region = region_list.back();
				region_list.pop_back();
				DrawRect(region,image);
			}
			display = image;
		}

		display.wait(100);
	}
}

void ShowRegions(const char* filename, const double low_threshold, const double high_threshold, const float sigma, const int min_area)
{
	CImg<double> image;

	if (strlen(filename))
	{
		image = CImg<>(filename);
		if (image.width() > kMaxImageWidth)
			image.resize(kResolution[0],kResolution[0]*image.height()/image.width());
	}
	else
	{
		image.resize(kResolution[0],kResolution[1]);
		image.load_camera(0,1,false,kResolution[0],kResolution[1]);
	}

	vector<KrabsRegion> region_list;
	KrabsLabeling(KrabsCanny(image.get_norm().normalize(0,255), sigma, low_threshold, high_threshold), region_list, min_area);
	while(!region_list.empty())
	{
		KrabsRegion region = region_list.back();
		region_list.pop_back();
		DrawRect(region, image);
	}

	image.display();
}

int main(int argc, char **argv)
{
	cimg_usage("Retrieve command line arguments");
	const char*  filename       = cimg_option("-i","","Input image file");
	const char   type           = cimg_option("-t",'m',"Algorithm type: e - Edge detection, b - Find button by Label, m = Motion detection");
	const double low_threshold  = cimg_option("-lt",15.0,"Low threshold");
	const double high_threshold = cimg_option("-ht",40.0,"High threshold");
	const float  sigma          = cimg_option("-s",1.4f,"Sigma");
	const char*  button_label   = cimg_option("-b","label","Button label");
	const int    min_area       = cimg_option("-a",5000,"Min area");
	const bool   show_threshold = cimg_option("-ts",false,"Show threshold");
	const int    dilate         = cimg_option("-d",20,"Dilate");

	try
	{
		switch(type)
		{
			case 'e':
			case 'E': EdgeDetection(filename, low_threshold, high_threshold, sigma); break;
			case 'b':
			case 'B': FindButton(filename, low_threshold, high_threshold, sigma, button_label, min_area); break;
			case 'm':
			case 'M': MotionDetection(sigma, min_area, high_threshold, dilate, show_threshold); break;
			case 'l':
			case 'L': ShowRegions(filename, low_threshold, high_threshold, sigma, min_area); break;
		}
	}
	catch(exception &ex)
	{
		std::cout << "Error:" << ex.what();
	}

	return 0;
}
