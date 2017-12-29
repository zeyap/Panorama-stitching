#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;

#define IMAGE_NUM 4
#define DOG_LAYER_NUM 4
#define PI 3.1416

struct Descriptor {
	vector<double> values;
	Point pos;
};

void Detect(Mat img, Mat & kps, vector<Point>& kpsIndex);
void FindExtremePoints(Mat * DoG, int layerNum, Mat & kps, vector<Point>& kpsIndex);
bool isCurrentPointExtreme(Mat * DoG, int layer, int row, int col);
bool isContrastHigh(Mat * DoG, int layer, int row, int col);
bool isCorner(Mat * DoG, int layer, int row, int col); 

void Describe(Mat img, vector<Point> kpsIndex, vector<Descriptor> & descriptors);
void DescribePoint(Mat img, int row, int col, vector<double> & descriptor);
void gradient(Mat img, int row, int col, double & m, double & o);
void binning(int windowIndex, vector<double> mag, vector<double> orient, vector<double>& bin);
int discretize(double angle);

int main(int argc, char ** argv)
{
	vector<vector<Descriptor>> descriptors;
	for (int i = 0; i < IMAGE_NUM; i++) {
		string fname = "yosemite"+to_string(i+1)+".jpg";
		Mat img_bgr = imread(fname);
		Mat img;
		cvtColor(img_bgr, img, CV_BGR2GRAY);
		Mat keypoints;
		vector<Point> kpsIndex;
		vector<Descriptor> newdescriptors;
		Detect(img,keypoints,kpsIndex);
		Describe(img,kpsIndex,newdescriptors);
		descriptors.push_back(newdescriptors);
	}

	for (int i = 0; i < IMAGE_NUM-1; i++) {
		Match(descriptors,i,i+1);
	}
	return 0;
}

void Detect(Mat img, Mat & kps, vector<Point>& kpsIndex) {

	vector<Mat> img_G;
	double sigma = 1.6;
	double k = sqrt(2);
	for (int i = 0; i < DOG_LAYER_NUM+1; i++) {
		double kernel=sigma;
		for (int j = 0; j < i; j++) {
			kernel *= k;
		}
		Mat g_dst;
		int ksz = (int)(kernel*2);
		if (ksz % 2 == 0) {
			ksz += 1;
		}
		GaussianBlur(img, g_dst,Size(ksz, ksz),0,0);
		img_G.push_back(g_dst);
	}

	Mat DoG[DOG_LAYER_NUM];
	for (int i = 0; i < DOG_LAYER_NUM; i++) {
		DoG[i] = img_G[i + 1] - img_G[i];
	}

	FindExtremePoints(DoG, DOG_LAYER_NUM, kps,kpsIndex);
	
}

void Describe(Mat img, vector<Point> kpsIndex, vector<Descriptor> & descriptors) {
	int kpsNum = kpsIndex.size();
	for (int i = 0; i < kpsNum; i++) {
		int r = kpsIndex[i].y;
		int c = kpsIndex[i].x;
		if ((r - 9 >= 0 && r + 9 < img.size().height) && (c - 9 >= 0 && c + 9 < img.size().width)) {
			Descriptor newdescriptor;
			DescribePoint(img, r, c, newdescriptor.values);
			newdescriptor.pos = kpsIndex[i];
			descriptors.push_back(newdescriptor);
		}
	}
}

void Match(vector<vector<Descriptor>> descriptors, int prev, int post) {

}

void DescribePoint(Mat img, int row, int col, vector<double> & descriptor) {
	
		vector<double> mag, orient;
		for (int r = row - 8; r < row + 8; r++) {
			for (int c = col - 8; c < col + 8; c++) {
				double newmag, neworient;
				gradient(img, r, c, newmag, neworient);
				mag.push_back(newmag);
				orient.push_back(neworient);
			}
		}
		for (int i = 0; i < 16; i++) {
			vector<double> bin;
			binning(i, mag, orient, bin);
			descriptor.insert(descriptor.end(), bin.begin(), bin.end());
		}
	
}

void gradient(Mat img, int row, int col, double & m, double & o) {
	int sobel[2][9] = { {-1,0,1,-2,0,2,-1,0,1}, {1,2,1,0,0,0,-1,-2,-1} };//x,y
	int yoffset[] = {-1,-1,-1,0,0,0,1,1,1};
	int xoffset[] = {-1,0,1,-1,0,1,-1,0,1};
	double g[2] = {0.0,0.0};
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 9; j++) {
			g[i] += img.at<uchar>(row+yoffset[j],col+xoffset[j])*sobel[i][j];
		}
	}
	m = sqrt(g[0] * g[0] + g[1] * g[1]);
	o = atan2(g[1],g[0]);//radians
}

void binning(int windowIndex, vector<double> mag, vector<double> orient, vector<double>& bin) {
	for (int i = 0; i < 8; i++) {
		bin.push_back(0);
	}
	int rowRangeL = (windowIndex/4)*4;//in 16x16 grid
	int rowRangeH = rowRangeL+3;
	int colRangeL = (windowIndex%4)*4;
	int colRangeH = colRangeL+3;
	for (int y = rowRangeL; y <= rowRangeH; y++) {
		for (int x = colRangeL; x <= colRangeH; x++) {
			int arrayIdx = y * 16 + x;
			double currAngle = orient[arrayIdx];
			bin[discretize(currAngle)]+=mag[arrayIdx];
		}
	}
}

int discretize(double angle) {
	//0-7 counterclockwise
	int res = (angle + PI / 8 + PI) / (PI / 4);
	return res>=4?(res-4):(res+4);
}

void FindExtremePoints(Mat * DoG, int layerNum, Mat & kps, vector<Point>& kpsIndex) {
	kps = Mat::zeros(DoG[0].size(), DoG[0].type());
	for (int layer = 1; layer <= layerNum - 2; layer++) {
		int h = DoG[layer].size().height;
		int w = DoG[layer].size().width;
		for (int i = 1; i < h - 1; i++) {
			for (int j = 1; j < w - 1; j++) {
				if (isCurrentPointExtreme(DoG, layer, i, j) 
					&& isContrastHigh(DoG, layer, i, j)) {
					kps.at<uchar>(i, j) = 255;
					kpsIndex.push_back(Point(j,i));
				}
			}
		}
	}
	imshow("kps",kps);
	waitKey();
}

bool isCurrentPointExtreme(Mat * DoG, int layer, int row, int col) {
	int r[] = {row-1,row-1,row-1,row,row,row,row+1,row+1,row+1};
	int c[] = {col-1,col,col+1,col-1,col,col+1,col-1,col,col+1};
	bool res = false;
	uchar temp = DoG[1].at<uchar>(row, col);
	for(int i=-1;i<=1;i++){
		for (int j = 0; j < 9; j++) {
			if (temp > DoG[layer + i].at<uchar>(r[j], c[j]))
				res = true;
		}
	}
	if (res == true) {
		return res;
	}
	for (int i = -1; i <= 1; i++) {
		for (int j = 0; j < 9; j++) {
			if (temp < DoG[layer + i].at<uchar>(r[j], c[j]))
				res = true;
		}
	}
	return res;
	
}

bool isContrastHigh(Mat * DoG, int layer, int row, int col) {
	uchar threshold = 15;
	if (DoG[layer].at<uchar>(row, col) > threshold) {
		return true;
	}
	else {
		return false;
	}

}

bool isCorner(Mat * DoG, int layer, int row, int col) {
	return true;
}
