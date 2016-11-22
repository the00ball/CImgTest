#include "krabs.h"

#define INBOUND_NW(x,y)     ((x) > 0 && (y) > 0)
#define INBOUND_NO(y)       ((y) > 0)
#define INBOUND_NE(img,x,y) ((x) < (img).width()-1 && (y) > 0)
#define INBOUND_EA(img,x)   ((x) < (img).width()-1)
#define INBOUND_SE(img,x,y) ((x) < (img).width()-1 && (y) < (img).height()-1)
#define INBOUND_SO(img,y)   ((y) < (img).height()-1)
#define INBOUND_SW(img,x,y) ((x) > 0 && (y) < (img).height()-1)
#define INBOUND_WE(x)       ((x) > 0)

#define NW(img,x,y) (img)((x)-1,(y)-1)
#define NO(img,x,y) (img)((x),(y)-1)
#define NE(img,x,y) (img)((x)+1,(y)-1)
#define EA(img,x,y) (img)((x)+1,(y))
#define SE(img,x,y) (img)((x)+1,(y)+1)
#define SO(img,x,y) (img)((x),(y)+1)
#define SW(img,x,y) (img)((x)-1,(y)+1)
#define WE(img,x,y) (img)((x)-1,(y))

CImg<double> KrabsSobel(const CImg<double>& gray)
{
	CImg<double> gradient_x = gray.get_convolve(kSobelKernelX).sqr().normalize(0, 255);
	CImg<double> gradient_y = gray.get_convolve(kSobelKernelY).sqr().normalize(0, 255);
	CImg<double> gradient   = ( gradient_x + gradient_y ).cut(0, 255).sqrt().normalize(0, 255);

	return gradient;
}

inline double ToDegrees(const double radians)
{
	return  (radians > 0 ? radians : radians + 2*M_PI) * 180/M_PI;
}

inline double AngleSum(const double angle, const double value)
{
	double result = angle + value;
	result = fmod(result, 360);
	return result < 0 ? result + 360 : result;
}

CImg<unsigned char> Hysteresis(const CImg<double> &gradient, const double high_threshold, const double low_threshold)
{
	vector<pair<int, int>> neighborhood;
	CImg<unsigned char> edge_trace = gradient.get_fill(0);

	cimg_forXY(gradient,x,y)
	{
		if (!edge_trace(x,y) && gradient(x,y) >= high_threshold)
		{
			edge_trace(x,y) = kEdge;

			CheckNeighborhood(neighborhood, gradient, edge_trace, x, y, low_threshold);

			while(!neighborhood.empty())
			{
				pair<int,int> point = neighborhood.back();
				neighborhood.pop_back();
				CheckNeighborhood(neighborhood, gradient, edge_trace, point.first, point.second, low_threshold);
			}
		}
	}

	return edge_trace;
}

inline void CheckNeighborhood(vector<pair<int, int>> &neighborhood, const CImg<double> &gradient, CImg<unsigned char> &edge_trace, const int x, const int y, const double threshold)
{
	// check 8-connected pixels

	if (x > 0 && y > 0 && !edge_trace(x-1,y-1) && gradient(x-1,y-1) >= threshold)
	{
		edge_trace(x-1,y-1) = kEdge;
		neighborhood.push_back(pair<int,int>(x-1,y-1));
	}

	if (y > 0 && !edge_trace(x,y-1) && gradient(x,y-1) >= threshold)
	{
		edge_trace(x,y-1) = kEdge;
		neighborhood.push_back(pair<int,int>(x,y-1));
	}

	if (x < gradient.width()-1 && y > 0 && !edge_trace(x+1,y-1) && gradient(x+1,y-1) >= threshold)
	{
		edge_trace(x+1,y-1) = kEdge;
		neighborhood.push_back(pair<int,int>(x+1,y-1));
	}

	if (x < gradient.width()-1 && !edge_trace(x+1,y) && gradient(x+1,y) >= threshold)
	{
		edge_trace(x+1,y) = kEdge;
		neighborhood.push_back(pair<int,int>(x+1,y));
	}

	if (x < gradient.width()-1 && y < gradient.height()-1 && !edge_trace(x+1,y+1) && gradient(x+1,y+1) >= threshold)
	{
		edge_trace(x+1,y+1) = kEdge;
		neighborhood.push_back(pair<int,int>(x+1,y+1));
	}

	if (y < gradient.height()-1 && !edge_trace(x,y+1) && gradient(x,y+1) >= threshold)
	{
		edge_trace(x,y+1) = kEdge;
		neighborhood.push_back(pair<int,int>(x,y+1));
	}

	if (x > 0 && y < gradient.height()-1 && !edge_trace(x-1,y+1) && gradient(x-1,y+1) >= threshold)
	{
		edge_trace(x-1,y+1) = kEdge;
		neighborhood.push_back(pair<int,int>(x-1,y+1));
	}

	if (x > 0 && !edge_trace(x-1,y) && gradient(x-1,y) >= threshold)
	{
		edge_trace(x-1,y) = kEdge;
		neighborhood.push_back(pair<int,int>(x-1,y));
	}
}

CImg<unsigned char> KrabsCanny(const CImg<double>& gray, const float sigma, const double low_threshold, const double high_threshold)
{
	// (1) Apply Gaussian filter to smooth the image in order to remove the noise

	CImg<double> gaussian = gray.get_blur(sigma, true, true);

	// (2) Find the intensity gradients of the image

	CImg<double> grad_x = gaussian.get_convolve(kSobelKernelX);
	CImg<double> grad_y = gaussian.get_convolve(kSobelKernelY);
	CImg<double> grad   = (grad_x.get_sqr().normalize(0,255)+grad_y.get_sqr().normalize(0,255) ).cut(0,255).sqrt().normalize(0,255);
	CImg<double> arc_tan2 = grad_y.get_atan2(grad_x);

	// (3) Apply non-maximum suppression to get rid of spurious response to edge detection

	const double kSector = 22.5;
	const double kAngles[4] = {0, 45, 90, 135};

	#pragma omp parallel for shared(grad,arc_tan2) schedule(dynamic,100)
	cimg_forXY(grad,x,y)
	{
		const double kArcTan2 = ToDegrees(arc_tan2(x,y));

		for (int i = 0; i < 4; i++)
		{
			const double kLowLimit1  = AngleSum(kAngles[i]    , -kSector);
			const double kHighLimit1 = AngleSum(kAngles[i]    ,  kSector);
			const double kLowLimit2  = AngleSum(kAngles[i]+180, -kSector);
			const double kHighLimit2 = AngleSum(kAngles[i]+180,  kSector);

			if ((!i &&
				(kArcTan2 > kLowLimit1 || kArcTan2 <= kHighLimit1)) ||
				(kArcTan2 > kLowLimit1 && kArcTan2 <= kHighLimit1) ||
				(kArcTan2 > kLowLimit2 && kArcTan2 <= kHighLimit2))
			{
				const double kGradientValue = grad(x,y);

				switch(static_cast<int>(kAngles[i]))
				{
					case 0:
						{
							if ((INBOUND_WE(x) && kGradientValue < WE(grad,x,y)) ||
								(INBOUND_EA(grad,x) && kGradientValue < EA(grad,x,y)))
								grad(x,y) = kSupress;
						}break;
					case 45:
						{
							if ((INBOUND_NE(grad,x,y) && kGradientValue < NE(grad,x,y)) ||
								(INBOUND_SW(grad,x,y) && kGradientValue < SW(grad,x,y)))
								grad(x,y) = kSupress;
						}break;
					case 90:
						{
							if ((INBOUND_NO(y) && kGradientValue < NO(grad,x,y)) ||
								(INBOUND_SO(grad,y) &&  kGradientValue < SO(grad,x,y)))
								grad(x,y) = kSupress;
						}break;
					case 135:
						{
							if ((INBOUND_NW(x,y) && kGradientValue < NW(grad,x,y)) ||
								(INBOUND_SE(grad,x,y) && kGradientValue < SE(grad,x,y)))
								grad(x,y) = kSupress;
						}break;
				}

				break;
			}
		}
	}

	// (4) Apply double threshold to determine potential edges
	// (5) Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

	return Hysteresis(grad, high_threshold, low_threshold);
}

inline void Labeling(vector<pair<int, int>> &neighborhood, const CImg<double> &binary, CImg<unsigned char> &labeled, KrabsRegion &region, const int x, const int y, const unsigned char current_label)
{
	// adjust label region

	region.x0 = x < region.x0 ? x : region.x0;
	region.x1 = x > region.x1 ? x : region.x1;

	region.y0 = y < region.y0 ? y : region.y0;
	region.y1 = y > region.y1 ? y : region.y1;

	// check 8-connected pixels

	if (x > 0 && y > 0 && !labeled(x-1,y-1) && binary(x-1,y-1))
	{
		labeled(x-1,y-1) = current_label;
		neighborhood.push_back(pair<int,int>(x-1,y-1));
	}

	if (y > 0 && !labeled(x,y-1) && binary(x,y-1))
	{
		labeled(x,y-1) = current_label;
		neighborhood.push_back(pair<int,int>(x,y-1));
	}

	if (x < binary.width()-1 && y > 0 && !labeled(x+1,y-1) && binary(x+1,y-1))
	{
		labeled(x+1,y-1) = current_label;
		neighborhood.push_back(pair<int,int>(x+1,y-1));
	}

	if (x < binary.width()-1 && !labeled(x+1,y) && binary(x+1,y))
	{
		labeled(x+1,y) = current_label;
		neighborhood.push_back(pair<int,int>(x+1,y));
	}

	if (x < binary.width()-1 && y < binary.height()-1 && !labeled(x+1,y+1) && binary(x+1,y+1))
	{
		labeled(x+1,y+1) = current_label;
		neighborhood.push_back(pair<int,int>(x+1,y+1));
	}

	if (y < binary.height()-1 && !labeled(x,y+1) && binary(x,y+1))
	{
		labeled(x,y+1) = current_label;
		neighborhood.push_back(pair<int,int>(x,y+1));
	}

	if (x > 0 && y < binary.height()-1 && !labeled(x-1,y+1) && binary(x-1,y+1))
	{
		labeled(x-1,y+1) = current_label;
		neighborhood.push_back(pair<int,int>(x-1,y+1));
	}

	if (x > 0 && !labeled(x-1,y) && binary(x-1,y))
	{
		labeled(x-1,y) = current_label;
		neighborhood.push_back(pair<int,int>(x-1,y));
	}
}

CImg<unsigned char> KrabsLabeling(const CImg<double> &binary, vector<KrabsRegion> &regions, const int min_area)
{
	const int kMaxArea = binary.width()*binary.height();

	vector<pair<int, int>> neighborhood;
	CImg<unsigned char> labeled = binary.get_fill(0);
	unsigned char current_label = 0;

	cimg_forXY(binary,x,y)
	{
		if (!labeled(x,y) && binary(x,y))
		{
			labeled(x,y) = ++current_label;

			KrabsRegion region;
			Labeling(neighborhood, binary, labeled, region, x, y, current_label);

			while(!neighborhood.empty())
			{
				pair<int,int> point = neighborhood.back();
				neighborhood.pop_back();
				Labeling(neighborhood, binary, labeled, region, point.first, point.second, current_label);
			}

			if (region.area() > min_area && region.area() < kMaxArea)
			{
				region.label = current_label;
				regions.push_back(region);
			}
		}
	}

	return labeled;
}

bool KrabsFindButton(const char* filename, vector<KrabsRegion> regions, const char* button_name, KrabsRegion& button_region)
{
	bool button_found = false;

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
			KrabsRegion region = regions.back();
			regions.pop_back();

			api->SetRectangle(region.x0, region.y0, region.width(), region.height());

			// Get OCR result
			string outText(api->GetUTF8Text());

			std::cout<<"OCR output: "<<outText<<"\n";

			if (outText.length() > 0 && outText.find(button_name) != string::npos)
			{
				button_region = region;
				button_found = true;
				break;
			}
		}

		// Destroy used object and release memory
		api->End();
		pixDestroy(&image);
	}

	return button_found;
}
