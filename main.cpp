#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace std;
using namespace cv;

#define IMAGE_NUM 4
#define DOG_LAYER_NUM 4
#define PI 3.1416

int main(int argc, char ** argv)
{
	vector<Mat> descriptors;
	vector<Mat> imgs;
	vector<vector<KeyPoint>> kps;
	for (int i = 0; i < IMAGE_NUM; i++) {
		string fname = "yosemite"+to_string(i+1)+".jpg";
		Mat img_bgr = imread(fname);
		Mat newimg;
		cvtColor(img_bgr, newimg, CV_BGR2GRAY);
		Ptr<Feature2D> f2d=xfeatures2d::SIFT::create();
		vector<KeyPoint> newkps;
		f2d->detect(newimg,newkps);
		Mat newdescriptor;
		f2d->compute(newimg,newkps,newdescriptor);

		descriptors.push_back(newdescriptor);
		imgs.push_back(newimg);
		kps.push_back(newkps);
	}

	BFMatcher matcher;
	for (int i = 0; i < IMAGE_NUM - 1; i++) {
		vector<DMatch> matches;
		Mat out;
		matcher.match(descriptors[i],descriptors[i+1],matches);
		drawMatches(imgs[i], kps[i], imgs[i + 1], kps[i + 1], matches, out);
		imshow("match",out);
		waitKey();
	}
	return 0;
}


