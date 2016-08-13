#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/video.hpp"
#include "opencv2/core.hpp"
#include <stdio.h>
#include <string>
#include <sstream>
#include <cmath>
#include <iostream>  

#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

using namespace sl::zed;
using namespace cv;
using namespace std;

#define UNKNOWN_FLOW_THRESH 1e9

//Camera variable and image dimensions
Camera* zed;
int width;
int height;

void makecolorwheel(vector<Scalar> &colorwheel)  
{  
    int RY = 15;  
    int YG = 6;  
    int GC = 4;  
    int CB = 11;  
    int BM = 13;  
    int MR = 6;  
  
    int i;  
  
    for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
}  


void motionToColor(cv::Mat flow, cv::Mat &color)  
{  
    if (color.empty())  
        color.create(flow.rows, flow.cols, CV_8UC3);  
  
    static vector<Scalar> colorwheel; //Scalar r,g,b  
    if (colorwheel.empty())  
        makecolorwheel(colorwheel);  
  
    // determine motion range:  
    float maxrad = -1;  
  
    // Find max flow to normalize fx and fy  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            float rad = sqrt(fx * fx + fy * fy);  
            maxrad = maxrad > rad ? maxrad : rad;  
        }  
    }  
  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
  
            float fx = flow_at_point[0] / maxrad;  
            float fy = flow_at_point[1] / maxrad;  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
            {  
                data[0] = data[1] = data[2] = 0;  
                continue;  
            }  
            float rad = sqrt(fx * fx + fy * fy);  
  
            float angle = atan2(-fy, -fx) / CV_PI;  
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
            int k0 = (int)fk;  
            int k1 = (k0 + 1) % colorwheel.size();  
            float f = fk - k0;  
            //f = 0; // uncomment to see original color wheel  
  
            for (int b = 0; b < 3; b++)   
            {  
                float col0 = colorwheel[k0][b] / 255.0;  
                float col1 = colorwheel[k1][b] / 255.0;  
                float col = (1 - f) * col0 + f * col1;  
                if (rad <= 1)  
                    col = 1 - rad * (1 - col); // increase saturation with radius  
                else  
                    col *= .75; // out of range  
                data[2 - b] = (int)(255.0 * col);  
            }  
        }  
    }  
}  




bool initCamera(){
	zed = new Camera(HD720, 30);

	InitParams param;
	param.mode = PERFORMANCE;
	param.unit = METER;
	param.verbose = 1;

	ERRCODE err = zed->init(param);
	cout << errcode2str(err)<<endl;
	if(err != SUCCESS){
		delete zed;
		return false;
	}

	//Image Size
	width = zed->getImageSize().width;
	height = zed->getImageSize().height;

	return true;
}


cv::Mat grabLeftFrame(){
	cv::Mat leftImage(height, width, CV_8UC4);
	sl::zed::Mat left = zed->retrieveImage(SIDE::LEFT);
	cv::Mat leftImage2(leftImage, cv::Rect(0, 0, width, height));
	sl::zed::slMat2cvMat(left).copyTo(leftImage2);
	
	return leftImage2;
}

cv::Mat grabRightFrame(){
	cv::Mat rightImage(height, width, CV_8UC4, 1);
	sl::zed::Mat right = zed->retrieveImage(SIDE::RIGHT);
	memcpy(rightImage.data, right.data, width*height*4*sizeof(uchar));
	//cv::cvtColor(rightImage, rightImage, cv::COLOR_BGRA2BGR);
	return rightImage;
}

cv::Mat grabDisparityMap(){
	cv::Mat depth(height, width, CV_8UC4, 1);
	sl::zed::Mat depthMap = zed->retrieveMeasure(sl::zed::MEASURE::DEPTH);
	memcpy(depth.data, depthMap.data, width*height*4*sizeof(uchar));
	return depth;
}

cv::Mat grabNormalizedDisparityMap(){
	cv::Mat normalizedDepth(height, width, CV_8UC4, 1);
	sl::zed::Mat normalizedDepthMap = zed->normalizeMeasure(sl::zed::MEASURE::DEPTH, 0.7, 20);
	return(slMat2cvMat(normalizedDepthMap));
}

int main(){
	cv::Mat left;

        cv::Mat prevgray, gray, flow, cflow, frame;  

        cv::Mat motion2color;
 
	

   if(initCamera()){

     for(;;)  
       {  
          if (zed->grab(SENSING_MODE::FILL))
             break;
          
          left = grabLeftFrame();
          double t = (double)cvGetTickCount();  
   
          cvtColor(left, gray, CV_BGR2GRAY);  
          imshow("original", left);  
  
          if( prevgray.data )  
           {  
            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);  
            motionToColor(flow, motion2color);  
            imshow("flow", motion2color);  
           }  
          if(waitKey(10)>=0)  
             break;  
          std::swap(prevgray, gray);  
  
          t = (double)cvGetTickCount() - t;  
          cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;  
        }
     }
     return 0;
}
