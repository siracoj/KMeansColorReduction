//CUDA Includes
#include "vector_types.h"

//OpenCV Includes
#include <cv.h>
#include <highgui.h>
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/stream_accessor.hpp"

//Base Includes
#include <stdlib.h> 


using namespace cv;

		

Mat color_img, output;
gpu::GpuMat d_src, d_dst; 
			

void kmeansInParallel_caller(uchar* src, uchar* dst, int width, int height);

void kmeansInParallel(const Mat& src, Mat& dst, int width, int height)
{	
	kmeansInParallel_caller(src.data, dst.data, width, height);
}


int main()
{
    int c;
	int height = 240;
	int width = 320;

    VideoCapture capture(0);
    namedWindow("Video",0); // create window

	if(!capture.isOpened())
		exit(-1);

    for(;;) {
		capture.read(color_img); // get frame
		if(!color_img.empty()){
			try{

				d_src.upload(color_img);

				kmeansInParallel(color_img,output, width, height); // run kmeans

				//d_dst.download(output); //copy data back

				imshow("Video", output); // show frame

				//d_dst.release();
				//d_src.release();

			}
			catch ( cv::Exception & e )
			{
				std::cout << e.msg << std::endl;
			}		
		}
        c = cvWaitKey(10); // wait 10 ms or for key stroke
        if(c == 27)
            break; // if ESC, break and quit
    }
    /* clean up */
	capture.release();
	capture.~VideoCapture();
}
