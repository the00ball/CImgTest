#include <CImg.h>
#include <omp.h>
#include <vector>
#include "lib/krabs.h"

using namespace std;
using namespace cimg_library;

void EdgeDetection(const char* filename, const double low_threshold, const double high_threshold, const float sigma)
{
	CImg<double> image;

	if (strcmp(filename,"cam.png"))
	{
		// load from file
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

	CImg<double> gray = image.get_norm().normalize(0,255);
	(image,KrabsSobel(gray),KrabsCanny(gray, sigma, low_threshold, high_threshold)).display();
}

void FindButton(const char* filename, const double low_threshold, const double high_threshold, const float sigma, const char* button_label)
{
	CImg<double> image;

	if (strcmp(filename,"cam.png"))
	{
		// load from file
		CImg<double> input(filename);
		image = input;
		if (image.width() > 640)
			image.resize(640, 640*image.height()/image.width());
	}
	else
	{
		image.resize(640,480);
		image.load_camera(0,0,false,640,480);
		FILE* file = cimg::fopen(filename, "wb+");
		image.save_png(file);
		fclose(file);
	}

	vector<KrabsRegion> region_list;
	CImg<unsigned int> labeled = KrabsLabeling(KrabsCanny(image.get_norm().normalize(0,255), sigma, low_threshold, high_threshold), region_list, 0);
	KrabsRegion button_region;
	if (KrabsFindButton(filename, region_list, button_label, button_region))
	{
		const double kColor[] = {0,255,0};
		image.draw_rectangle(button_region.x0,button_region.y0,button_region.x1,button_region.y1,kColor,0.2f);
	}

	image.display();
}

//! Motion Detection
/**
 *  Based: http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
 */
void MotionDetection(const float sigma, const int mim_area, const double high_threshold, const int dilate, const bool show_threshold)
{
	const double kGreen[] = {0,255,0};
	const double kRed[] = {255,0,0};

	CImg<double> image(640,480);
	CImg<double> first_frame(640,480);

	vector<KrabsRegion> region_list;
	CImgDisplay display(image, "Motion Detection");

	first_frame.load_camera(0,1,false,640,480).norm().normalize(0,255).blur(sigma,true,true);
	display.display(image);

	while(!display.is_closed())
	{
		image.load_camera(0,0,false,640,480);
		CImg<double> gray = image.get_norm().normalize(0,255).blur(sigma,true,true);
		CImg<double> threshold = (first_frame - gray).abs().threshold(high_threshold).dilate(dilate);

		if (show_threshold)
			display = threshold;
		else
		{
			KrabsLabeling(threshold, region_list, mim_area);
			while(!region_list.empty())
			{
				KrabsRegion region = region_list.back();
				region_list.pop_back();
				image.draw_rectangle(region.x0, region.y0, region.x1, region.y1, kGreen, 0.2f);
				image.draw_line(region.x0, region.y0, region.x0, region.y1, kGreen, 1);
				image.draw_line(region.x0, region.y0, region.x1, region.y0, kGreen, 1);
				image.draw_line(region.x1, region.y0, region.x1, region.y1, kGreen, 1);
				image.draw_line(region.x0, region.y1, region.x1, region.y1, kGreen, 1);
				image.draw_text((region.x0+region.x1)/2, (region.y0+region.y1)/2, "X", kRed);
			}
			display = image;
		}

		display.wait(100);
	}
}

int main(int argc, char **argv)
{
	cimg_usage("Retrieve command line arguments");
	const char*  filename       = cimg_option("-i","cam.png","Input image file");
	const char   type           = cimg_option("-t",'m',"Algorithm type: e - Edge detection, b - Find button by Label, m = Motion detection");
	const double low_threshold  = cimg_option("-lt",15.0,"Low threshold");
	const double high_threshold = cimg_option("-ht",40.0,"High threshold");
	const float  sigma          = cimg_option("-s",1.4f,"Sigma");
	const char*  button_label   = cimg_option("-b","","Button label");
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
			case 'B': FindButton(filename, low_threshold, high_threshold, sigma, button_label); break;
			case 'm':
			case 'M': MotionDetection(sigma, min_area, high_threshold, dilate, show_threshold); break;
		}
	}
	catch(exception &ex)
	{
		std::cout << "Error:" << ex.what();
	}

	return 0;
}
