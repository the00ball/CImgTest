# CImgTest

The CImgTest is a simple project to test the CImg Library. It has just only one .cpp file CImgEdgeDetection.cpp. As the name says, this program applies edge detection over an input source image. To cover more than one edge detection approaches, two diferents techinics are developed:

* [Sobel](https://en.wikipedia.org/wiki/Sobel_operator)  
* [Canny](https://en.wikipedia.org/wiki/Canny_edge_detector)

As result of this process, the images below represents the output of this technics as well.

## Sobel
![Sobel Operator](https://github.com/the00ball/CImgTest/blob/master/img/lena_sobel.png?raw=true)

## Canny
![Canny Algorithm](https://github.com/the00ball/CImgTest/blob/master/img/lena_canny.png?raw=true)

## Enviroment

This repository is pre-configured for Eclipse CDT enviroment: Eclipse Neon. 1a Release(4.6.1).

## Dependencies

* Linux enviroment(Developed over Ubuntu 16.10)
* Eclipse Neon. 1a Release(4.6.1) or more recent version;
* CDT enviroment;
* CImg source file (CImg.h) 1.7.8
* Jpeg lib packages;
* OpenCV lib packages;

## Compilation

The project uses the built-in g++ under linux distribution.

### Command Line

g++ -o edgedetection CImgEdgeDetection.cpp -O2 -Dcimg_display=1 -Dcimg_use_jpeg -Dcimg_use_opencv -L/usr/X11R6/lib -I/path/to/cimg -I/usr/include/opencv -lm -lpthread -lX11 -l:libjpeg.so.8 -lopencv_core -lopencv_highgui

More details in [CImg Web Site](http://cimg.eu/reference/group__cimg__overview.html)

## Testing the Program

If you run it with no command line options, just doing ./edgedetection, the program will use the default options below:

* Image source: main camera (For snapshot, the CImg library use OpenCV with a plugin mechanism, se more in [CImg.eu](http://cimg.eu/))
* Algorithm: canny
* Low threshold: 15 
* High threshold: 60

## Retrieving Command Line Options

Just type in your terminal ./edgedetection --help and the follow lines will be shown:

edgedetection: Retrieve command line arguments (Nov  8 2016, 10:16:06)

    -i                                        Input image file
    -t               C                        Algorithm type: (S)obel or (C)anny
    -lt              15                       Low threshold
    -ht              60                       High threshold
    

