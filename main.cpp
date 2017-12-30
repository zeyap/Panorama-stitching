#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

#define IMAGE_NUM 4

void Log(vector<DMatch> matches, string fname);
void SaveImage(string name, Mat img);
void FilterMatchesByDistance(vector<DMatch> & matches, double distThres);
void Fit_Ransac(vector<DMatch> matches, vector<KeyPoint> queryKps, vector<KeyPoint> trainKps, int seedSz, int iterNum, Mat & homography);
bool Fit(vector<DMatch> matches, vector<KeyPoint> dstKps, vector<KeyPoint> srcKps, vector<int>shuffleArray, int seedSz, Mat & homography);

void DetectAndDescribe(vector<Mat> & imgs_bgr, vector<Mat> & imgs, vector<Mat> & descriptors, vector<vector<KeyPoint>> & kps);
void MatchNextTo(int i, BFMatcher matcher, vector<DMatch> & matches, vector<Mat> imgs, vector<Mat> descriptors, vector<vector<KeyPoint>> kps);
void WarpNextToFit(int i, vector<DMatch> & matches, vector<Mat> & imgs_bgr, vector<vector<KeyPoint>> kps, Mat & lasthomo, vector<Mat> & warps);
void Stitch(vector<Mat> warps, Mat & dst);

int main(int argc, char ** argv)
{
	vector<Mat> descriptors;
	vector<Mat> imgs; 
	vector<Mat> imgs_bgr;
	vector<vector<KeyPoint>> kps;

	DetectAndDescribe(imgs_bgr, imgs, descriptors, kps);

	vector<Mat> warps;

	BFMatcher matcher;

	Mat lasthomo;
	for (int i = 0; i < IMAGE_NUM - 1; i++) {
		vector<DMatch> matches;
		MatchNextTo(i, matcher, matches, imgs, descriptors, kps);
		WarpNextToFit(i, matches,imgs_bgr, kps,lasthomo, warps);
	}

	Mat stitch;
	Stitch(warps,stitch);
	SaveImage("stitch", stitch);
	waitKey();

	return 0;
}

void DetectAndDescribe(vector<Mat> & imgs_bgr, vector<Mat> & imgs, vector<Mat> & descriptors, vector<vector<KeyPoint>> & kps) {
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(0, 4, 0.1, 10, 1.6);
	for (int i = 0; i < IMAGE_NUM; i++) {
		string fname = "yosemite" + to_string(i + 1) + ".jpg";
		Mat img_bgr = imread(fname);
		imgs_bgr.push_back(img_bgr);
		Mat newimg;
		cvtColor(img_bgr, newimg, CV_BGR2GRAY);
		vector<KeyPoint> newkps;
		f2d->detect(newimg, newkps);
		Mat newdescriptor;
		f2d->compute(newimg, newkps, newdescriptor);
		//cout << newkps[0].pt << endl;
		descriptors.push_back(newdescriptor);
		imgs.push_back(newimg);
		kps.push_back(newkps);
	}
}

void MatchNextTo(int i, BFMatcher matcher, vector<DMatch> & matches, vector<Mat> imgs, vector<Mat> descriptors, vector<vector<KeyPoint>> kps) {
	Mat out;
	matcher.match(descriptors[i + 1], descriptors[i], matches);
	FilterMatchesByDistance(matches, 100);
	string match_fname = "match" + to_string(i) + "_" + to_string(i + 1);
	Log(matches, match_fname);
	drawMatches(imgs[i + 1], kps[i + 1], imgs[i], kps[i], matches, out);
	SaveImage(match_fname, out);
}

void WarpNextToFit(int i, vector<DMatch> & matches, vector<Mat> & imgs_bgr, vector<vector<KeyPoint>> kps, Mat & lasthomo, vector<Mat> & warps) {
	Mat newhomo;
	//i+1->i
	Fit_Ransac(matches, kps[i + 1], kps[i], 3, 10, newhomo);
	Mat warpRes;
	if (i > 0) {
		newhomo *= lasthomo;
	}
	lasthomo = newhomo.clone();

	if (i == 0) {
		warps.push_back(imgs_bgr[i]);
	}
	warpPerspective(imgs_bgr[i + 1], warpRes, newhomo, Size(imgs_bgr[i].size().width*(i + 2), imgs_bgr[i + 1].size().height));
	warps.push_back(warpRes);
	SaveImage("warp" + to_string(i + 1), warpRes);
}

void SaveImage(string name,Mat img) {
	imwrite("output/"+name+".png",img);
}

bool FileExist(string str) {
	string newStr = str;
	std::ifstream fin(newStr);
	if (fin) {
		cout << "File " + str + " already exists" << endl;
		return true;
	}
	else {
		return false;
	}
}

void Log(vector<DMatch> matches, string fname) {
	fname = "output/" + fname+".txt";
	if (FileExist(fname)) {
		return;
	}
	ofstream outfile(fname);
	
	int sz = matches.size();
	outfile << sz <<endl;
	for (int i = 0; i < sz;i++) {
		outfile << matches[i].queryIdx<<'\t';
		outfile << matches[i].trainIdx << '\t';
		outfile << matches[i].distance << endl;
	}

	outfile.close();
}

void FilterMatchesByDistance(vector<DMatch> & matches, double distThres) {
	vector<DMatch> newMatch;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance <= distThres) {
			newMatch.push_back(matches[i]);
		}
	}
	matches.swap(newMatch);
}

void Fit_Ransac(vector<DMatch> matches, vector<KeyPoint> queryKps, vector<KeyPoint> trainKps,int seedSz, int iterNum, Mat & homography) {
	vector<int> shuffleArray;
	for (int i = 0; i < matches.size(); i++) {
		shuffleArray.push_back(i);
	}

	for (int iteration = 1; iteration <= iterNum; iteration++) {
		random_shuffle(shuffleArray.begin(), shuffleArray.end());
		//calculate affine transform query-(affine trans)->train //size(query)>size(train)
		if (Fit(matches, trainKps,queryKps,shuffleArray,seedSz,homography) == true) {
			cout << "good fittings found" << endl;
			break;
		}
	}
}

bool Fit(vector<DMatch> matches, vector<KeyPoint> dstKps, vector<KeyPoint> srcKps,vector<int>shuffleArray, int seedSz, Mat & homography) { //seedSz>=3
	bool res = true;

	vector<Point2f> dstpts;
	vector<Point2f> srcpts;

	for (int i = 0; i < seedSz; i++) {
		int shuffleIndex = shuffleArray[i];
		srcpts.push_back(srcKps[matches[shuffleIndex].queryIdx].pt);
		dstpts.push_back(dstKps[matches[shuffleIndex].trainIdx].pt);
	}
	
	Mat warp = getAffineTransform(srcpts, dstpts);

	int inlierNum = 0;
	double inlierthres = 20;
	double minInlierRate = 0.8;
	
	for (int i = 0; i < matches.size(); i++) {
		int srcIdx=matches[i].queryIdx;
		int dstIdx = matches[i].trainIdx;
		double x = srcKps[srcIdx].pt.x;
		double y = srcKps[srcIdx].pt.y;
		double x_dst = dstKps[dstIdx].pt.x;
		double y_dst = dstKps[dstIdx].pt.y;
		double x_a = warp.at<double>(0, 0)*x + warp.at<double>(0,1)*y+ warp.at<double>(0, 2);
		double y_a = warp.at<double>(1, 0)*x + warp.at<double>(1, 1)*y + warp.at<double>(1, 2);
		double dist = sqrt((x_a-x_dst)*(x_a - x_dst)+ (y_a - y_dst)*(y_a - y_dst));
		if (dist <= inlierthres) {
			inlierNum++;
		}
	}

	if (inlierNum < minInlierRate * matches.size()) {
		res = false;
	}
	homography = Mat::zeros(Size(3, 3), CV_64F);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i <= 1) {
				homography.at<double>(i, j) = warp.at<double>(i, j);
			}
			else if (j == 2) {
				homography.at<double>(i, j) = 1;
			}
		}
	}
	cout << homography.at<double>(0, 0) << "\t" << homography.at<double>(0, 1) << "\t" << homography.at<double>(0, 2) << endl;
	cout << homography.at<double>(1, 0) << "\t" << homography.at<double>(1, 1) << "\t" << homography.at<double>(1, 2) << endl;
	cout << homography.at<double>(2, 0) << "\t" << homography.at<double>(2, 1) << "\t" << homography.at<double>(2, 2) << endl;
	
	return res;
}

void Stitch(vector<Mat> warps,Mat & dst) {
	int sz = warps.size();
	dst = warps[sz - 1].clone();

	int h = warps[sz-1].size().height;
	int w = warps[sz-1].size().width;

	for (int i = sz-2; i >=0; i--) {
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < warps[i].size().width; x++) {
				if (dst.at<Vec3b>(y, x) == Vec3b(0,0,0) && warps[i].at<Vec3b>(y, x) != Vec3b(0, 0, 0)) {
					dst.at<Vec3b>(y, x) = warps[i].at<Vec3b>(y, x);
				}
			}
		}
	}
}