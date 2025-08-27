#include <iostream>
#include <xtgmath.h>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <opencv2\core\types.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2\stitching\detail\camera.hpp>
#include <opencv2\stitching\detail\autocalib.hpp>
#include <opencv2\stitching\detail\motion_estimators.hpp>
#include "dirent-master\include\dirent.h"
#include "Transform.h"
#include "BundleAdjustment.h"
#include "eigen3\Eigen\SVD"
#include "eigen3/Eigen/IterativeLinearSolvers"
#include "vlfeat-0.9.20/vl/sift.h"

//#define IMAGEDEBUG

#define	OPENCV_SIFT 0
#define	VLFEAT_SIFT 1


using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace Eigen;

//struct Matches
//{
//	vector<int> inliers_match1, inliers_match2;
//};

struct VerticesInfo
{
	int num_x, num_y;
	float dw, dh;
};

struct MatchingPointsandIndex
{
	vector<vector<Point2f>> points;//all matching points for each image
	vector<vector<int>> img_index;//matchinfo indices of all matching points for each image
	vector<vector<int>> class_index;//plane class indices of all matching points for each image
};

vector<string> getImageFileFullNamesInDir(const string & dir_name) {
	DIR *dir;
	struct dirent *ent;
	vector<string> result;

	const vector<string> image_formats = {
		".bmp", ".dib",
		".jpeg", ".jpg", ".jpe", ".JPG",
		".jp2",
		".png", ".PNG"
		".pbm", ".pgm", ".ppm",
		".sr", ".ras",
		".tiff", ".tif" };

	if ((dir = opendir(dir_name.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			string file = string(ent->d_name);
			for (int i = 0; i < image_formats.size(); ++i) {
				if (file.length() > image_formats[i].length() &&
					image_formats[i].compare(file.substr(file.length() - image_formats[i].length(),
						image_formats[i].length())) == 0) {
					result.emplace_back(dir_name + file);
				}
			}
		}
		closedir(dir);
	}
	return result;
}

void featureDetect(const Mat &_grey_img,
	vector<Point2f> & _feature_points,
	Mat & _feature_descriptors)
{
#if OPENCV_SIFT
	Ptr<cv::xfeatures2d::SIFT> siftDetector = cv::xfeatures2d::SIFT::create(0, 3, 0.01, 50, 1.6);
	vector<KeyPoint> key_points;
	siftDetector->detectAndCompute(_grey_img, noArray(), key_points, _feature_descriptors);
	KeyPoint::convert(key_points, _feature_points);
#elif VLFEAT_SIFT
	vector<Mat> feature_descriptors;
	cv::Mat grey_img_float = _grey_img.clone();
	grey_img_float.convertTo(grey_img_float, CV_32FC1);

	const int  width = _grey_img.cols;
	const int height = _grey_img.rows;

	VlSiftFilt * vlSift = vl_sift_new(width, height,
		log2(min(width, height)),
		3,
		0);
	vl_sift_set_peak_thresh(vlSift, 0.0);
	vl_sift_set_edge_thresh(vlSift, 500.0);

	if (vl_sift_process_first_octave(vlSift, (vl_sift_pix const *)grey_img_float.data) != VL_ERR_EOF) {
		do {
			vl_sift_detect(vlSift);
			for (int i = 0; i < vlSift->nkeys; ++i) {
				double angles[4];


				int angleCount = vl_sift_calc_keypoint_orientations(vlSift, angles, &vlSift->keys[i]);
				for (int j = 0; j < angleCount; ++j) {
					cv::Mat descriptor_array(1, 128, CV_32FC1);
					vl_sift_calc_keypoint_descriptor(vlSift, (vl_sift_pix *)descriptor_array.data, &vlSift->keys[i], angles[j]);
					_feature_points.push_back(Point2f(vlSift->keys[i].x, vlSift->keys[i].y));
					feature_descriptors.push_back(descriptor_array);
				}

			}
		} while (vl_sift_process_next_octave(vlSift) != VL_ERR_EOF);
	}
	vl_sift_delete(vlSift);

	_feature_descriptors.create(feature_descriptors.size(), 128, CV_32FC1);
	for (int i = 0; i < feature_descriptors.size(); i++)
	{
		feature_descriptors[i].copyTo(_feature_descriptors.row(i));
	}
#endif
}

void featureMatch(const Mat &discriptor1, const Mat &discriptor2, vector<int>& matchIdxs1, vector<int>& matchIdxs2)
{
	FlannBasedMatcher matcher;
	vector<vector<DMatch> > matchePoints;
	vector<DMatch> GoodMatchePoints;

	vector<Mat> train_desc(1, discriptor2);
	matcher.add(train_desc);
	matcher.train();

	matcher.knnMatch(discriptor1, matchePoints, 2);
	//matcher.radiusMatch(imageDesc2, matchePoints, 0.1);
	//matcher.match(imageDesc2, GoodMatchePoints);
	//cout << "total match points: " << matchePoints.size() << endl;

	// Lowe's algorithm
	for (int i = 0; i < matchePoints.size(); i++)
	{
		if (1.22 * matchePoints[i][0].distance < matchePoints[i][1].distance)
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
		}
	}

	matchIdxs1.resize(GoodMatchePoints.size());
	matchIdxs2.resize(GoodMatchePoints.size());
	for (size_t i = 0; i < GoodMatchePoints.size(); i++)
	{
		matchIdxs1[i] = GoodMatchePoints[i].queryIdx;
		matchIdxs2[i] = GoodMatchePoints[i].trainIdx;
	}
}

template <typename T>
void pointsDistance(const Point_<T> p, const vector<Point_<T>>& p2, vector<double>& dis)
{
	for (int i = 0; i < p2.size(); i++)
	{
		double d = std::sqrt(double((p.x - p2[i].x)*(p.x - p2[i].x) + (p.y - p2[i].y)*(p.y - p2[i].y)));
		dis.push_back(d);
	}
}

Mat mdlt(const vector<Point2f> &cf1, const vector<Point2f> &cf2, const vector<double> &dis, const double sigma)
{
	MatrixXf A = MatrixXf::Zero(cf1.size() * 2, 9);
	double gamma = 0.0025;
	//(cv::max)(Gki, Mat(Gki.size(), Gki.type(), Scalar(gamma)), Wi);

	for (int j = 0; j < cf1.size(); ++j)
	{
		double wi = max(exp(-dis[j] / sigma), gamma);
		A(2 * j, 0) = wi * cf1[j].x;
		A(2 * j, 1) = wi * cf1[j].y;
		A(2 * j, 2) = wi * 1;
		A(2 * j, 6) = wi * -cf2[j].x * cf1[j].x;
		A(2 * j, 7) = wi * -cf2[j].x * cf1[j].y;
		A(2 * j, 8) = wi * -cf2[j].x;

		A(2 * j + 1, 3) = wi * cf1[j].x;
		A(2 * j + 1, 4) = wi * cf1[j].y;
		A(2 * j + 1, 5) = wi * 1;
		A(2 * j + 1, 6) = wi * -cf2[j].y * cf1[j].x;
		A(2 * j + 1, 7) = wi * -cf2[j].y * cf1[j].y;
		A(2 * j + 1, 8) = wi * -cf2[j].y;
	}
	JacobiSVD<MatrixXf, HouseholderQRPreconditioner> jacobi_svd(A, ComputeThinV);
	MatrixXf V = jacobi_svd.matrixV();
	cv::Mat H(3, 3, CV_64FC1);
	for (int j = 0; j < V.rows(); ++j) {
		H.at<double>(j / 3, j % 3) = V(j, V.rows() - 1);
	}
	return H;
}

template <typename T>
void deleteVector(vector<T> &v, const vector<int> &idx)
{
	vector<T> v1;
	vector<int> bidx(v.size(), 0);
	for (int i = 0; i < idx.size(); i++)
	{
		bidx[idx[i]] = 1;
	}
	for (int i = 0; i < v.size(); i++)
	{
		if (!bidx[i])
			v1.push_back(v[i]);
	}
	v = v1;
}

void pointsRansac(const vector<Point2f> &points1, const vector<Point2f> &points2,
	const vector<int> &match1_in, const vector<int> &match2_in,
	/*vector<Point2f> &points1_out, vector<Point2f> &points2_out,*/
	MatchInfo &matchinfo)
{
	vector<Point2f> re_points1, re_points2;//matching points by ransac
	vector<int> match1_out, match2_out;
	vector<int> class_idx;
	int class_num = 0;
	vector<Mat> class_H;
	
	vector<Point2f> tmp_points1, tmp_points2;
	tmp_points1.assign(points1.begin(), points1.end());
	tmp_points2.assign(points2.begin(), points2.end());
	vector<int> tmp_match1, tmp_match2;
	tmp_match1.assign(match1_in.begin(), match1_in.end());
	tmp_match2.assign(match2_in.begin(), match2_in.end());
	int max_inliers_count = 0;
	for (int i = 0; i < 20; ++i)
	{
		Mat inliers;
		Mat H;
		H = findHomography(tmp_points1, tmp_points2, CV_RANSAC, 3.0, inliers, 500);
		uchar* pinliers = inliers.ptr<uchar>(0);
		int inliers_count = 0;
		int inliers_count2 = 0;
		for (int j = 0; j < tmp_points1.size(); ++j)
		{
			if (pinliers[j])
			{
				++inliers_count;
				if (PointsTransformDistance(tmp_points2[j], tmp_points1[j], H.inv()) <= 3.0)
				{
					++inliers_count2;
				}
			}

		}
		if (inliers_count < 15)
			break;
		vector<Point2f> next_points1, next_points2;
		vector<int> next_match1, next_match2;
		if (inliers_count2 < 15)
		{
			for (int j = 0; j < tmp_points1.size(); ++j)
			{
				if (!pinliers[j])
				{
					next_points1.push_back(tmp_points1[j]);
					next_points2.push_back(tmp_points2[j]);
					next_match1.push_back(tmp_match1[j]);
					next_match2.push_back(tmp_match2[j]);
				}
			}
			tmp_points1 = next_points1;
			tmp_points2 = next_points2;
			tmp_match1 = next_match1;
			tmp_match2 = next_match2;
			i--;
			continue;
		}
		if (max_inliers_count == 0)
			max_inliers_count = inliers_count;

		Scalar clr = Scalar_<uchar>(rand() % 256, 0, rand() % 256);
		vector<Point2f> in_points1, in_points2;
		vector<int> in_match1, in_match2;
		for (int j = 0; j < tmp_points1.size(); ++j)
		{
			if (pinliers[j])
			{
				in_points1.push_back(tmp_points1[j]);
				in_points2.push_back(tmp_points2[j]);
				in_match1.push_back(tmp_match1[j]);
				in_match2.push_back(tmp_match2[j]);
				class_idx.push_back(i);
			}
			else
			{
				next_points1.push_back(tmp_points1[j]);
				next_points2.push_back(tmp_points2[j]);
				next_match1.push_back(tmp_match1[j]);
				next_match2.push_back(tmp_match2[j]);
			}
		}
		tmp_points1 = next_points1;
		tmp_points2 = next_points2;
		tmp_match1 = next_match1;
		tmp_match2 = next_match2;
		re_points1.insert(re_points1.end(), in_points1.begin(), in_points1.end());
		re_points2.insert(re_points2.end(), in_points2.begin(), in_points2.end());
		match1_out.insert(match1_out.end(), in_match1.begin(), in_match1.end());
		match2_out.insert(match2_out.end(), in_match2.begin(), in_match2.end());

		if (inliers_count >= 0.05*max_inliers_count && inliers_count >= 50)
		{
			//	vector<Point2f> center;
			class_H.push_back(H);
			//	center.push_back(calcCenter(in_points1));
			//	center.push_back(calcCenter(in_points2));
			//	class_center.push_back(center);
		}
		//class_H.push_back(H);
		class_num++;
	}

	if (re_points1.size() == 0 || re_points2.size() == 0)
	{
		matchinfo.isUse = false;
		return;
	}
		

	vector<Point2f> nf1, nf2, cf1, cf2;
	Mat T1 = getNormalize2DPts(re_points1, nf1);
	Mat T2 = getNormalize2DPts(re_points2, nf2);
	Mat C1 = getConditionerFromPts(nf1);
	Mat C2 = getConditionerFromPts(nf2);
	cf1.reserve(nf1.size());
	cf2.reserve(nf2.size());
	for (int i = 0; i < nf1.size(); ++i) {
		cf1.emplace_back(nf1[i].x * C1.at<double>(0, 0) + C1.at<double>(0, 2),
			nf1[i].y * C1.at<double>(1, 1) + C1.at<double>(1, 2));

		cf2.emplace_back(nf2[i].x * C2.at<double>(0, 0) + C2.at<double>(0, 2),
			nf2[i].y * C2.at<double>(1, 1) + C2.at<double>(1, 2));
	}
	//MatrixXf A = MatrixXf::Zero(cf1.size() * 2, 9);

	vector<int> outliers;
	for (int i = 0; i < re_points1.size(); ++i)
	{
		vector<double>dis;
		pointsDistance(re_points1[i], re_points1, dis);

		vector<int> sort_idx;
		sortIdx(Mat(dis).t(), sort_idx, SORT_EVERY_ROW + SORT_ASCENDING);

		vector<int> num20(class_num, 0);
		for (int j = 0; j < 20; j++)
		{
			num20[class_idx[sort_idx[j]]]++;
		}
		int max_cnt_class_idx = 0;
		for (int k = 0; k < class_num; k++)
		{
			if (num20[k] > num20[max_cnt_class_idx])
				max_cnt_class_idx = k;
		}
		if (class_idx[i] == max_cnt_class_idx)
			continue;
		
		double sigma = dis[sort_idx[19]];
		if (sigma > 100.0)
			sigma = 100.0;
		else if (sigma < 40.0)
			sigma = 40.0;

		Mat H = mdlt(cf1, cf2, dis, sigma);

		H = C2.inv() * H * C1;
		H = T2.inv() * H * T1;

		if ((PointsTransformDistance(re_points1[i], re_points2[i], H) > 6.0
			|| PointsTransformDistance(re_points2[i], re_points1[i], H.inv()) > 6.0))
			outliers.push_back(i);
	}

	deleteVector(re_points1, outliers);
	deleteVector(re_points2, outliers);
	deleteVector(match1_out, outliers);
	deleteVector(match2_out, outliers);
	deleteVector(class_idx, outliers);
	vector<Point2f> main_points1, main_points2;
	vector<int> main_class_idx;
	for (int i = 0; i < class_idx.size(); ++i)
	{
		if (class_idx[i] < class_H.size())
		{
			main_points1.push_back(re_points1[i]);
			main_points2.push_back(re_points2[i]);
			main_class_idx.push_back(class_idx[i]);
		}
	}
	//int last_num = -1;
	//for (int i = 0; i < class_idx.size(); i++)
	//{
	//	if (i == class_idx.size() - 1)
	//	{
	//		std::cout << i - last_num << endl;
	//		last_num = i;
	//	}
	//	else if (class_idx[i] != class_idx[i + 1])
	//	{
	//		std::cout << i - last_num << endl;
	//		last_num = i;
	//	}

	//}
	matchinfo.isUse = true;
	matchinfo.H = findHomography(re_points1, re_points2);
	matchinfo.class_H = class_H;
	matchinfo.class_idx = class_idx;
	matchinfo.class_num = class_num;
	matchinfo.inliers_match1 = match1_out;
	matchinfo.inliers_match2 = match2_out;
	matchinfo.re_points1 = re_points1;
	matchinfo.re_points2 = re_points2;
	matchinfo.main_points1 = main_points1;
	matchinfo.main_points2 = main_points2;
	matchinfo.main_class_idx = main_class_idx;
}

template <typename T>
void getMedianWithoutCopyData(vector<T> & _vec, double & _median) {
	size_t n = _vec.size() / 2;
	std::nth_element(_vec.begin(), _vec.begin() + n, _vec.end());
	_median = _vec[n];
	if ((_vec.size() & 1) == 0) {
		std::nth_element(_vec.begin(), _vec.begin() + n - 1, _vec.end());
		_median = (_median + _vec[n - 1]) * 0.5;
	}
}

double estimateFocal(const vector<Mat> &images, const vector<MatchInfo> &info)
{
	vector<Mat> translation_matrix;
	translation_matrix.reserve(images.size());
	for (int i = 0; i < images.size(); ++i) {
		Mat T(3, 3, CV_64F);
		T.at<double>(0, 0) = T.at<double>(1, 1) = T.at<double>(2, 2) = 1;
		T.at<double>(0, 2) = images[i].cols * 0.5;
		T.at<double>(1, 2) = images[i].rows * 0.5;
		T.at<double>(0, 1) = T.at<double>(1, 0) = T.at<double>(2, 0) = T.at<double>(2, 1) = 0;
		translation_matrix.emplace_back(T);
	}

	vector<double> image_focal_candidates;
	for (int i = 0; i < info.size(); i++)
	{
		if (info[i].isUse == false)
			continue;
		for (int j = 0; j < info[i].class_H.size(); j++)
		{
			Mat h = translation_matrix[info[i].img_idx2].inv() * info[i].class_H[j] * translation_matrix[info[i].img_idx2];
			double f0, f1;
			bool f0_ok, f1_ok;
			detail::focalsFromHomography(h / h.at<double>(2, 2), f0, f1, f0_ok, f1_ok);
			if (f0_ok && f1_ok)
			{
				image_focal_candidates.push_back(f0);
				image_focal_candidates.push_back(f1);
			}
		}

	}
	double focal;
	getMedianWithoutCopyData(image_focal_candidates, focal);
	return focal;
}

int findStitchingOrder(const int image_num, const vector<MatchInfo> &info, vector<Vec2i> &order)
{
	vector<int> collect_num(image_num, 0);
	vector<int> points_num(image_num, 0);
	for (int i = 0; i < info.size(); i++)
	{
		if (info[i].isUse)
		{
			collect_num[info[i].img_idx1]++;
			collect_num[info[i].img_idx2]++;
			points_num[info[i].img_idx1] += info[i].inliers_match1.size();
			points_num[info[i].img_idx2] += info[i].inliers_match2.size();
		}
	}
	int center_img_idx = 0;
	list<int> remain_img;
	for (int i = 0; i < image_num; i++)
	{
		if (collect_num[i] > collect_num[center_img_idx])
			center_img_idx = i;
		else if(collect_num[i] == collect_num[center_img_idx]&& points_num[i]>points_num[center_img_idx])
			center_img_idx = i;
		remain_img.push_back(i);
	}
	list<int>::iterator it = remain_img.begin();
	advance(it, center_img_idx);
	remain_img.erase(it);
	order.push_back(Vec2i(center_img_idx, -1));
	while (remain_img.size() != 0)
	{
		int max_points_num = 0;
		int next_idx;
		int next_idx_f;
		list<int>::const_iterator next_it;
		for (list<int>::const_iterator iter = remain_img.begin(); iter != remain_img.end(); iter++)
		{
			for (int j = 0; j < order.size(); j++)
			{
				int k1 = (std::min)((*iter), order[j][0]);
				int k2 = (std::max)((*iter), order[j][0]);
				int pair_idx = (2 * image_num - k1 - 1)*k1 / 2 + k2 - k1 - 1;
				if (info[pair_idx].inliers_match1.size() > max_points_num)
				{
					max_points_num = info[pair_idx].inliers_match1.size();
					next_idx = (*iter);
					next_idx_f = order[j][0];
					next_it = iter;
				}
			}
		}
		order.push_back(Vec2i(next_idx, next_idx_f));
		remain_img.erase(next_it);
	}



	return center_img_idx;
}

void projectPoints1(const vector<Point2f> &src_points, vector<Point2f> &dst_points, const Mat &H)
{
	for (int i = 0; i < src_points.size(); i++)
	{
		dst_points.push_back(applyTransform3x3(src_points[i].x, src_points[i].y, H));
	}
}

void seperatePointsSet(const vector<Point2f> &src_points, const vector<int> &idx, vector<vector<Point2f>> &dst_points)
{
	dst_points.resize(idx[idx.size() - 1] + 1);
	for (int i = 0; i < idx.size(); i++)
	{
		dst_points[idx[i]].push_back(src_points[i]);
	}
}

Mat estimateNormal(const vector<Point2f> &points1, const vector<Point2f> &points2, const Mat &R, const Mat &t)
{
	Mat A = Mat::zeros(points1.size() * 2, 9, CV_64F);
	for (int i = 0; i < points1.size(); ++i)
	{
		A.at<double>(2 * i, 0) = points1[i].x;
		A.at<double>(2 * i, 1) = points1[i].y;
		A.at<double>(2 * i, 2) = 1.0;
		A.at<double>(2 * i, 6) = -points2[i].x * points1[i].x;
		A.at<double>(2 * i, 7) = -points2[i].x * points1[i].y;
		A.at<double>(2 * i, 8) = -points2[i].x;

		A.at<double>(2 * i + 1, 3) = points1[i].x;
		A.at<double>(2 * i + 1, 4) = points1[i].y;
		A.at<double>(2 * i + 1, 5) = 1.0;
		A.at<double>(2 * i + 1, 6) = -points2[i].y * points1[i].x;
		A.at<double>(2 * i + 1, 7) = -points2[i].y * points1[i].y;
		A.at<double>(2 * i + 1, 8) = -points2[i].y;
	}
	Mat Rvec(9, 1, CV_64F);
	for (int j = 0; j < 9; ++j)
	{
		Rvec.at<double>(j) = R.at<double>(j / 3, j % 3);
	}	
	Mat T = Mat::zeros(9, 3, CV_64F);
	T.at<double>(0, 0) = T.at<double>(1, 1) = T.at<double>(2, 2) = t.at<double>(0, 0);
	T.at<double>(3, 0) = T.at<double>(4, 1) = T.at<double>(5, 2) = t.at<double>(1, 0);
	T.at<double>(6, 0) = T.at<double>(7, 1) = T.at<double>(8, 2) = t.at<double>(2, 0);
	Mat W = A * T;
	Mat b = -A * Rvec;
	Mat x;
	solve(W, b, x, DECOMP_NORMAL);
	return x.t();
}

double computeEpipolarError(const vector<Point2f> &points1, const vector<Point2f> &points2, const double focal, const Mat &E)
{
	double err = 0.0;
	for (int i = 0; i < points1.size(); ++i)
	{
		Mat x1(3, 1, CV_64F);
		x1.at<double>(0, 0) = points1[i].x / focal;
		x1.at<double>(1, 0) = points1[i].y / focal;
		x1.at<double>(2, 0) = 1.0;
		Mat x2(3, 1, CV_64F);
		x2.at<double>(0, 0) = points2[i].x / focal;
		x2.at<double>(1, 0) = points2[i].y / focal;
		x2.at<double>(2, 0) = 1.0;
		Mat Ex1 = E * x1;
		Mat Etx2 = E.t()*x2;
		Mat x2tEx1 = x2.t()*Ex1;
		double x2tEx11 = x2.dot(Ex1);
		err += sqrt((x2tEx1.at<double>(0, 0)*x2tEx1.at<double>(0, 0))
			/ (Ex1.at<double>(0, 0)*Ex1.at<double>(0, 0) + Ex1.at<double>(1, 0)*Ex1.at<double>(1, 0)
				+ Etx2.at<double>(0, 0)*Etx2.at<double>(0, 0) + Etx2.at<double>(1, 0)*Etx2.at<double>(1, 0)))*focal;

	}
	return err;
}

void calcEssentialMat(const vector<Mat> &images, vector<MatchInfo> &info, const vector<Vec2i> &order, double& focal)
{
	for (int ii = 1; ii < order.size(); ii++)
	{
		int k1 = (std::min)(order[ii][0], order[ii][1]);
		int k2 = (std::max)(order[ii][0], order[ii][1]);
		int i = (2 * images.size() - k1 - 1)*k1 / 2 + k2 - k1 - 1;

		Mat K0 = Mat::eye(3, 3, CV_64F);
		//K0.at<double>(0, 0) = focal;
		//K0.at<double>(1, 1) = focal;
		K0.at<double>(0, 2) = images[info[i].img_idx1].cols * 0.5;
		K0.at<double>(1, 2) = images[info[i].img_idx1].rows * 0.5;

		Mat K1 = Mat::eye(3, 3, CV_64F);
		//K1.at<double>(0, 0) = focal;
		//K1.at<double>(1, 1) = focal;
		K1.at<double>(0, 2) = images[info[i].img_idx2].cols * 0.5;
		K1.at<double>(1, 2) = images[info[i].img_idx2].rows * 0.5;

		vector<Point2f> points1, points2;
		projectPoints1(info[i].re_points1, points1, K0.inv());
		projectPoints1(info[i].re_points2, points2, K1.inv());

		double min_error = 100000.0;
		double correct_focal;
		for (double j = 0.5; j <= 2.0; j += 0.05)
		{
			Mat mask;
			Mat E = findEssentialMat(points1, points2, focal * j, Point2d(0, 0), RANSAC, 0.999, 1.0, mask);
			double err = computeEpipolarError(points1, points2, focal * j, E);

			if (err < min_error)
			{
				min_error = err;
				correct_focal = j * focal;
			}
		}
		focal = correct_focal;
		Mat correct_E;
		Mat correct_mask;
		min_error = 100000.0;
		for (double j = 0.8; j <= 1.2; j += 0.01)
		{
			Mat mask;
			int num = 0;
			Mat E = findEssentialMat(points1, points2, focal * j, Point2d(0, 0), RANSAC, 0.999, 1.0, mask);
			double err = computeEpipolarError(points1, points2, focal * j, E);

			if (err < min_error)
			{
				min_error = err;
				correct_focal = j * focal;
				correct_E = E;
				correct_mask = mask;
			}
		}
		cout<<(float)min_error / points1.size()<<","<< correct_focal <<endl;
		focal = correct_focal;

		recoverPose(correct_E, points1, points2, info[i].R, info[i].t, correct_focal, Point2d(0, 0), correct_mask);
	}
}

void calcNormalVector(vector<MatchInfo> &info, const vector<detail::CameraParams> &cameras)
{
	for (int i = 0; i < info.size(); i++)
	{
		if (!info[i].isUse) continue;
		Mat K0 = Mat::eye(3, 3, CV_64F);
		K0.at<double>(0, 0) = cameras[info[i].img_idx1].focal;
		K0.at<double>(1, 1) = cameras[info[i].img_idx1].focal;
		K0.at<double>(0, 2) = cameras[info[i].img_idx1].ppx;
		K0.at<double>(1, 2) = cameras[info[i].img_idx1].ppy;

		Mat K1 = Mat::eye(3, 3, CV_64F);
		K1.at<double>(0, 0) = cameras[info[i].img_idx2].focal;
		K1.at<double>(1, 1) = cameras[info[i].img_idx2].focal;
		K1.at<double>(0, 2) = cameras[info[i].img_idx2].ppx;
		K1.at<double>(1, 2) = cameras[info[i].img_idx2].ppy;

		vector<Point2f> points1, points2;
		projectPoints1(info[i].re_points1, points1, K0.inv());
		projectPoints1(info[i].re_points2, points2, K1.inv());
		vector<vector<Point2f>> pts1, pts2;
		seperatePointsSet(points1, info[i].class_idx, pts1);
		seperatePointsSet(points2, info[i].class_idx, pts2);

		Mat R = cameras[info[i].img_idx2].R*cameras[info[i].img_idx1].R.inv();
		Mat t = cameras[info[i].img_idx2].t - R * cameras[info[i].img_idx1].t;
		for (int j = 0; j < pts1.size(); j++)
		{
			Mat n = Mat::zeros(1, 3, CV_64F);
			if (pts1[j].size() >= 10)
				n = estimateNormal(pts1[j], pts2[j], R, t);
			info[i].class_normal.push_back(n);
			Mat fH = K1 * (R + t*n)*K0.inv();
		}
		for (int j = 0; j < info[i].class_normal.size(); j++)
		{
			info[i].class_normal[j] = info[i].class_normal[j] * cameras[info[i].img_idx1].R;
		}
	}
}

Mat Rotation2Homography(const Mat &image, const double focal, const Mat &R)
{
	Mat_<float> K = Mat::eye(3, 3, CV_32F);
	K(0, 0) = focal;
	K(1, 1) = focal;
	K(0, 2) = image.cols * 0.5;
	K(1, 2) = image.rows * 0.5;
	Mat H1 = K * R.inv() * K.inv();
	return H1.inv();
}

void estimateCameras(const vector<Mat> &images, const vector<MatchInfo> &info, const vector<Vec2i> &order, const double focal, vector<detail::CameraParams> &cameras)
{
	for (int i = 0; i < order.size(); i++)
	{
		cameras[order[i][0]].focal = focal;
		cameras[order[i][0]].aspect = 1.0;
		cameras[order[i][0]].ppx = images[order[i][0]].cols / 2;
		cameras[order[i][0]].ppy = images[order[i][0]].rows / 2;
		if (i == 0)
		{
			cameras[order[i][0]].R = Mat::eye(3, 3, CV_64F);
			cameras[order[i][0]].t = Mat::zeros(3, 1, CV_64F);
		}
		else
		{
			int k1 = (std::min)(order[i][0], order[i][1]);
			int k2 = (std::max)(order[i][0], order[i][1]);
			int pair_idx = (2 * images.size() - k1 - 1)*k1 / 2 + k2 - k1 - 1;
			if (k1 == order[i][0])
			{
				cameras[order[i][0]].R = info[pair_idx].R.inv()*cameras[order[i][1]].R;
				cameras[order[i][0]].t = info[pair_idx].R.inv()*(cameras[order[i][1]].t - info[pair_idx].t);
			}
			else
			{
				cameras[order[i][0]].R = info[pair_idx].R*cameras[order[i][1]].R;
				cameras[order[i][0]].t = info[pair_idx].R*cameras[order[i][1]].t + info[pair_idx].t;
			}
		}

	}
}

void compromiseCameras(vector<detail::CameraParams> &cameras)
{
	Mat rvec = Mat::zeros(3, 1, CV_64F);
	Mat tvec = Mat::zeros(3, 1, CV_64F);
	for (int i = 0; i < cameras.size(); i++)
	{
		Mat r;
		Rodrigues(cameras[i].R, r);
		rvec += r;
		tvec += cameras[i].t;
	}
	rvec /= cameras.size();
	tvec /= cameras.size();
	Mat R;
	Rodrigues(rvec, R);
	for (int i = 0; i < cameras.size(); i++)
	{
		cameras[i].R = cameras[i].R*R.inv();
		cameras[i].t = -cameras[i].R*tvec + cameras[i].t;
	}
}

void image_warping(const Mat &image, Mat &warped_image, const vector<vector<Mat>> &Hs, float cw, float ch, float dw, float dh, float min_x, float min_y)
{
	warped_image.create(cvCeil(ch), cvCeil(cw), image.type());
	for (int i = 0; i < warped_image.rows; i++)
	{
		Vec3b* pdata = warped_image.ptr<Vec3b>(i);
		for (int j = 0; j < warped_image.cols; j++)
		{
			Mat H = Hs[cvRound(j / dw)][cvRound(i / dh)];
			H.convertTo(H, CV_64F);
			Point2f pt = applyTransform3x3(j + min_x, i + min_y, H);
			if (pt.x >= 0 && pt.x < image.cols&&pt.y >= 0 && pt.y < image.rows)
				pdata[j] = image.at<Vec3b>(Point((int)pt.x, (int)pt.y));
			else
				pdata[j] = Vec3b(0, 0, 0);
		}
	}
}

void image_warping(const Mat &image, Mat &warped_image, const Mat &_H, float cw, float ch, float min_x, float min_y)
{
	warped_image.create(cvCeil(ch), cvCeil(cw), image.type());
	Mat H;
	_H.convertTo(H, CV_64F);
	for (int i = 0; i < warped_image.rows; i++)
	{
		Vec3b* pdata = warped_image.ptr<Vec3b>(i);
		for (int j = 0; j < warped_image.cols; j++)
		{
			Point2f pt = applyTransform3x3(j + min_x, i + min_y, H.inv());
			if (pt.x >= 0 && pt.x < image.cols&&pt.y >= 0 && pt.y < image.rows)
				pdata[j] = image.at<Vec3b>(Point((int)pt.x, (int)pt.y));
			else
				pdata[j] = Vec3b(0, 0, 0);
		}
	}
}

void image_blending(vector<Mat> &images, Mat &result)
{
	result = Mat::zeros(images[0].size(), images[0].type());
	for (int i = 0; i < images[0].rows; i++)
	{
		vector<Vec3b*> pdatas(images.size());
		for (int n = 0; n < images.size(); n++)
		{
			pdatas[n] = images[n].ptr<Vec3b>(i);
		}
		Vec3b* presult = result.ptr<Vec3b>(i);
		for (int j = 0; j < images[0].cols; j++)
		{
			int b_value = 0, g_value = 0, r_value = 0;
			int num = 0;
			for (int n = 0; n < images.size(); n++)
			{
				if (pdatas[n][j][0] != 0 || pdatas[n][j][1] != 0 || pdatas[n][j][2] != 0)
				{
					b_value += pdatas[n][j][0];
					g_value += pdatas[n][j][1];
					r_value += pdatas[n][j][2];
					num++;
				}
			}
			if (num > 0)
			{
				presult[j][0] = b_value / num;
				presult[j][1] = g_value / num;
				presult[j][2] = r_value / num;
			}
			else
				presult[j] = Vec3b(0, 0, 0);
		}
	}
}

void BundleAdjustment::setUpInitialCameraParams(const std::vector<CameraParams> &cameras, const vector<MatchInfo> &info)
{
	for (int i = 0; i < info.size(); i++)
	{
		num_planes_ += info[i].class_normal.size();
	}
	all_params_.create(num_images_ * 7 + num_planes_ * 3, 1, CV_64F);
	cam_params_ = all_params_(Rect(0, 0, 1, num_images_ * 7));
	plane_params_ = all_params_(Rect(0, num_images_ * 7, 1, num_planes_ * 3));
	//cam_params_.create(num_images_ * 7, 1, CV_64F);
	SVD svd;
	for (int i = 0; i < num_images_; ++i)
	{
		cam_params_.at<double>(i * 7, 0) = cameras[i].focal;
		Mat rvec;
		Rodrigues(cameras[i].R, rvec);
		CV_Assert(rvec.type() == CV_64F);
		cam_params_.at<double>(i * 7 + 1, 0) = rvec.at<double>(0, 0);
		cam_params_.at<double>(i * 7 + 2, 0) = rvec.at<double>(1, 0);
		cam_params_.at<double>(i * 7 + 3, 0) = rvec.at<double>(2, 0);
		cam_params_.at<double>(i * 7 + 4, 0) = cameras[i].t.at<double>(0, 0);
		cam_params_.at<double>(i * 7 + 5, 0) = cameras[i].t.at<double>(1, 0);
		cam_params_.at<double>(i * 7 + 6, 0) = cameras[i].t.at<double>(2, 0);
	}
	//plane_params_.create(num_planes_ * 3, 1, CV_64F);
	int num = 0;
	for (int i = 0; i < info.size(); i++)
	{
		for (int j = 0; j < info[i].class_normal.size(); j++)
		{
			Mat n = info[i].class_normal[j]/* * cameras[info[i].img_idx1].R*/;
			plane_params_.at<double>(num * 3, 0) = n.at<double>(0, 0);
			plane_params_.at<double>(num * 3 + 1, 0) = n.at<double>(0, 1);
			plane_params_.at<double>(num * 3 + 2, 0) = n.at<double>(0, 2);
			num++;
		}
	}
	

}

void BundleAdjustment::obtainRefinedCameraParams(std::vector<CameraParams> &cameras, vector<MatchInfo> &info)
{
	for (int i = 0; i < num_images_; ++i)
	{
		cameras[i].focal = cam_params_.at<double>(i * 7, 0);

		Mat rvec(3, 1, CV_64F);
		rvec.at<double>(0, 0) = cam_params_.at<double>(i * 7 + 1, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(i * 7 + 2, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(i * 7 + 3, 0);
		Rodrigues(rvec, cameras[i].R);

		Mat tvec(3, 1, CV_64F);
		tvec.at<double>(0, 0) = cam_params_.at<double>(i * 7 + 4, 0);
		tvec.at<double>(1, 0) = cam_params_.at<double>(i * 7 + 5, 0);
		tvec.at<double>(2, 0) = cam_params_.at<double>(i * 7 + 6, 0);
		cameras[i].t = tvec;
	}
	int num = 0;
	for (int i = 0; i < info.size(); i++)
	{
		for (int j = 0; j < info[i].class_normal.size(); j++)
		{
			Mat n(1, 3, CV_64F);			
			n.at<double>(0, 0) = plane_params_.at<double>(num * 3, 0);
			n.at<double>(0, 1) = plane_params_.at<double>(num * 3 + 1, 0);
			n.at<double>(0, 2) = plane_params_.at<double>(num * 3 + 2, 0);
			info[i].class_normal[j] = n;
			num++;
		}
	}
}

void BundleAdjustment::calcError(Mat &err)
{
	err.create(total_num_matches_ * 3, 1, CV_64F);

	int match_idx = 0;
	int plane_idx = 0;
	for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx)
	{
		int i = edges_[edge_idx].first;
		int j = edges_[edge_idx].second;
		double f1 = cam_params_.at<double>(i * 7, 0);
		double f2 = cam_params_.at<double>(j * 7, 0);

		double R1[9];
		Mat R1_(3, 3, CV_64F, R1);
		Mat rvec(3, 1, CV_64F);
		rvec.at<double>(0, 0) = cam_params_.at<double>(i * 7 + 1, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(i * 7 + 2, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(i * 7 + 3, 0);
		Rodrigues(rvec, R1_);
		Mat t1(3, 1, CV_64F);
		t1.at<double>(0, 0) = cam_params_.at<double>(i * 7 + 4, 0);
		t1.at<double>(1, 0) = cam_params_.at<double>(i * 7 + 5, 0);
		t1.at<double>(2, 0) = cam_params_.at<double>(i * 7 + 6, 0);

		double R2[9];
		Mat R2_(3, 3, CV_64F, R2);
		rvec.at<double>(0, 0) = cam_params_.at<double>(j * 7 + 1, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(j * 7 + 2, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(j * 7 + 3, 0);
		Rodrigues(rvec, R2_);
		Mat t2(3, 1, CV_64F);
		t2.at<double>(0, 0) = cam_params_.at<double>(j * 7 + 4, 0);
		t2.at<double>(1, 0) = cam_params_.at<double>(j * 7 + 5, 0);
		t2.at<double>(2, 0) = cam_params_.at<double>(j * 7 + 6, 0);

		const MatchInfo& matches_info = info_[(2 * num_images_ - i - 1)*i / 2 + j - i - 1];

		for (size_t k = 0; k < matches_info.re_points1.size(); ++k)
		{
			Point2f p1 = matches_info.re_points1[k];
			int idx = plane_idx + matches_info.class_idx[k];
			Mat N(3, 1, CV_64F);
			N.at<double>(0, 0) = plane_params_.at<double>(idx * 3, 0);
			N.at<double>(1, 0) = plane_params_.at<double>(idx * 3 + 1, 0);
			N.at<double>(2, 0) = plane_params_.at<double>(idx * 3 + 2, 0);
			Point2f p11;
			p11.x = (p1.x - cameras_[i].ppx) / f1;
			p11.y = (p1.y - cameras_[i].ppy) / f1;
			Mat H1 = (R1_ + t1 * N.t()).inv();
			double x1 = H1.at<double>(0, 0)*p11.x + H1.at<double>(0, 1)*p11.y + H1.at<double>(0, 2);
			double y1 = H1.at<double>(1, 0)*p11.x + H1.at<double>(1, 1)*p11.y + H1.at<double>(1, 2);
			double z1 = H1.at<double>(2, 0)*p11.x + H1.at<double>(2, 1)*p11.y + H1.at<double>(2, 2);
			double len = std::sqrt(x1*x1 + y1 * y1 + z1 * z1);
			x1 /= len; y1 /= len; z1 /= len;

			Point2f p2 = matches_info.re_points2[k];
			Point2f p22;
			p22.x = (p2.x - cameras_[j].ppx) / f2;
			p22.y = (p2.y - cameras_[j].ppy) / f2;
			Mat H2 = (R2_ + t2 * N.t()).inv();
			double x2 = H2.at<double>(0, 0)*p22.x + H2.at<double>(0, 1)*p22.y + H2.at<double>(0, 2);
			double y2 = H2.at<double>(1, 0)*p22.x + H2.at<double>(1, 1)*p22.y + H2.at<double>(1, 2);
			double z2 = H2.at<double>(2, 0)*p22.x + H2.at<double>(2, 1)*p22.y + H2.at<double>(2, 2);
			len = std::sqrt(x2*x2 + y2 * y2 + z2 * z2);
			x2 /= len; y2 /= len; z2 /= len;

			double mult = std::sqrt(f1 * f2);
			err.at<double>(3 * match_idx, 0) = mult * (x1 - x2);
			err.at<double>(3 * match_idx + 1, 0) = mult * (y1 - y2);
			err.at<double>(3 * match_idx + 2, 0) = mult * (z1 - z2);

			match_idx++;
		}
		plane_idx += matches_info.class_normal.size();
	}
}

void calcDeriv(const Mat &err1, const Mat &err2, double h, Mat res)
{
	for (int i = 0; i < err1.rows; ++i)
		res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}

void BundleAdjustment::calcJacobian(Mat &jac)
{
	jac.create(total_num_matches_ * 3, num_images_ * 7 + num_planes_ * 3, CV_64F);

	double val;
	double step = 1e-3;

	for (int i = 0; i < num_images_; ++i)
	{
		for (int j = 1; j < 7; ++j)
		{
			val = cam_params_.at<double>(i * 7 + j, 0);
			cam_params_.at<double>(i * 7 + j, 0) = val - step;
			Mat err1_, err2_;
			calcError(err1_);
			cam_params_.at<double>(i * 7 + j, 0) = val + step;
			calcError(err2_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + j));
			cam_params_.at<double>(i * 7 + j, 0) = val;
		}
	}
	step = 1e-5;
	for (int i = 0; i < num_planes_; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			val = plane_params_.at<double>(i * 3 + j, 0);
			plane_params_.at<double>(i * 3 + j, 0) = val - step;
			Mat err1_, err2_;
			calcError(err1_);
			plane_params_.at<double>(i * 3 + j, 0) = val + step;
			calcError(err2_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(num_images_ * 7 + i * 3 + j));
			plane_params_.at<double>(i * 3 + j, 0) = val ;
		}
	}
}

bool BundleAdjustment::estimate(const vector<vector<Point2f>> &all_keypoints, vector<MatchInfo> &info, vector<detail::CameraParams> &cameras)
{
	num_images_ = static_cast<int>(all_keypoints.size());
	all_keypoints_ = &all_keypoints[0];
	info_ = &info[0];
	cameras_ = &cameras[0];

	setUpInitialCameraParams(cameras, info);

	edges_.clear();
	for (int i = 0; i < num_images_ - 1; ++i)
	{
		for (int j = i + 1; j < num_images_; ++j)
		{			
			if (info_[(2 * num_images_ - i - 1)*i / 2 + j - i - 1].isUse)
				edges_.push_back(std::make_pair(i, j));
		}
	}

	total_num_matches_ = 0;
	for (size_t i = 0; i < edges_.size(); ++i)
	{
		int pair_idx = (2 * num_images_ - edges_[i].first - 1)*edges_[i].first / 2 + edges_[i].second - edges_[i].first - 1;
		total_num_matches_ += static_cast<int>(info[pair_idx].re_points1.size());
	}
	CvLevMarq solver(num_images_ * num_params_per_cam_ + num_planes_*3,
		total_num_matches_ * num_errs_per_measurement_,
		term_criteria_);
	Mat err, jac;
	CvMat matParams = all_params_;
	cvCopy(&matParams, solver.param);
	int iter = 0;
	for (;;)
	{
		const CvMat* _param = 0;
		CvMat* _jac = 0;
		CvMat* _err = 0;

		bool proceed = solver.update(_param, _jac, _err);

		cvCopy(_param, &matParams);

		if (!proceed || !_err)
			break;

		if (_jac)
		{
			calcJacobian(jac);
			CvMat tmp = jac;
			cvCopy(&tmp, _jac);
		}

		if (_err)
		{
			calcError(err);
			cout << std::sqrt(err.dot(err) / total_num_matches_)<<endl;
			if (std::sqrt(err.dot(err) / total_num_matches_) < 1.5|| iter>30) break;
			iter++;
			CvMat tmp = err;
			cvCopy(&tmp, _err);
		}
	}
	obtainRefinedCameraParams(cameras, info);
	return true;
}

void LeastSquaresSolve(vector<Triplet<double> > & _triplets,const vector<pair<int, double> > & _b_vector, 
	vector<vector<Point2f>> &proj_vertices, int equations, int verticesCount)
{
	LeastSquaresConjugateGradient<SparseMatrix<double> > lscg;
	SparseMatrix<double> A(equations, verticesCount*2);
	VectorXd b = VectorXd::Zero(equations), x;
	A.setFromTriplets(_triplets.begin(), _triplets.end());
	for (int i = 0; i < _b_vector.size(); ++i) {
		b[_b_vector[i].first] = _b_vector[i].second;
	}
	lscg.compute(A);
	x = lscg.solve(b);
	int k = 0;
	for (int i = 0; i < proj_vertices.size(); i++)
	{
		proj_vertices[i].resize(verticesCount / proj_vertices.size());
		for (int j = 0; j < proj_vertices[i].size(); j++)
		{
			proj_vertices[i][j].x = x[k];
			proj_vertices[i][j].y = x[k + 1];
			k += 2;
		}
	}
}

void prepareAlignmentTerm(vector<Triplet<double> > & _triplets, const vector<VerticesInfo> &vertices, const vector<MatchInfo> &info)
{	
	int N = vertices[0].num_x*vertices[0].num_y;
	int num_equ = 0;
	for (int i = 0; i < info.size(); i++)
	{
		for (int j = 0; j < info[i].re_points1.size(); j++)
		{
			for(int k=0;k<2;k++)
			{
				Point2f pt1;
				int img_idx;
				if (k == 0)
				{
					pt1 = info[i].re_points1[j];
					img_idx = info[i].img_idx1;
				}
				else
				{
					pt1 = info[i].re_points2[j];
					img_idx = info[i].img_idx2;
				}				
				VerticesInfo v1 = vertices[img_idx];
				double w1 = (cvCeil(pt1.x / v1.dw)*v1.dw - pt1.x)*(cvCeil(pt1.y / v1.dh)*v1.dh - pt1.y);
				double w2 = (cvCeil(pt1.x / v1.dw)*v1.dw - pt1.x)*(pt1.y - cvFloor(pt1.y / v1.dh)*v1.dh);
				double w3 = (pt1.x - cvFloor(pt1.x / v1.dw)*v1.dw)*(pt1.y - cvFloor(pt1.y / v1.dh)*v1.dh);
				double w4 = (pt1.x - cvFloor(pt1.x / v1.dw)*v1.dw)*(cvCeil(pt1.y / v1.dh)*v1.dh - pt1.y);
				double sw = (w1 + w2 + w3 + w4);
				w1 /= sw;
				w2 /= sw;
				w3 /= sw;
				w4 /= sw;
				if (k == 1)
				{
					w1 = -w1; w2 = -w2; w3 = -w3; w4 = -w4;
				}
				int v_idx1 = cvFloor(pt1.x / v1.dw)*v1.num_y + cvFloor(pt1.y / v1.dh);
				int v_idx2 = cvFloor(pt1.x / v1.dw)*v1.num_y + cvCeil(pt1.y / v1.dh);
				int v_idx3 = cvCeil(pt1.x / v1.dw)*v1.num_y + cvCeil(pt1.y / v1.dh);
				int v_idx4 = cvCeil(pt1.x / v1.dw)*v1.num_y + cvFloor(pt1.y / v1.dh);
				_triplets.emplace_back(num_equ * 2, (v_idx1 + img_idx*N) * 2, w1);
				_triplets.emplace_back(num_equ * 2 + 1, (v_idx1 + img_idx * N) * 2 + 1, w1);
				_triplets.emplace_back(num_equ * 2, (v_idx2 + img_idx *N) * 2, w2);
				_triplets.emplace_back(num_equ * 2 + 1, (v_idx2 + img_idx * N) * 2 + 1, w2);
				_triplets.emplace_back(num_equ * 2, (v_idx3 + img_idx *N) * 2, w3);
				_triplets.emplace_back(num_equ * 2 + 1, (v_idx3 + img_idx * N) * 2 + 1, w3);
				_triplets.emplace_back(num_equ * 2, (v_idx4 + img_idx *N) * 2, w4);
				_triplets.emplace_back(num_equ * 2 + 1, (v_idx4 + img_idx * N) * 2 + 1, w4);
			}
			num_equ++;

		}
	}
}

void prepareSimilarityTerm(vector<Triplet<double> > & _triplets, vector<pair<int, double> > & _b_vector, 
	const vector<detail::CameraParams> &cameras, const vector<VerticesInfo> &vertices, int AlignEquNum)
{
	int N = vertices[0].num_x*vertices[0].num_y;
	int num_equ = AlignEquNum;
	Mat K0 = Mat::eye(3, 3, CV_64F);
	K0.at<double>(0, 0) = cameras[0].focal;
	K0.at<double>(1, 1) = cameras[0].focal;
	K0.at<double>(0, 2) = cameras[0].ppx;
	K0.at<double>(1, 2) = cameras[0].ppy;
	for (int i = 0; i < vertices.size(); i++)
	{
		Mat K = Mat::eye(3, 3, CV_64F);
		K.at<double>(0, 0) = cameras[i].focal;
		K.at<double>(1, 1) = cameras[i].focal;
		K.at<double>(0, 2) = cameras[i].ppx;
		K.at<double>(1, 2) = cameras[i].ppy;
		Mat tx = Mat::zeros(3, 3, CV_64F);
		tx.at<double>(0, 1) = tx.at<double>(1, 0) = cameras[i].t.at<double>(2, 0);
		tx.at<double>(0, 2) = cameras[i].t.at<double>(1, 0);
		tx.at<double>(2, 0) = -cameras[i].t.at<double>(1, 0);
		tx.at<double>(2, 1) = cameras[i].t.at<double>(0, 0);
		tx.at<double>(1, 2) = -cameras[i].t.at<double>(0, 0);
		Mat E = K.inv().t()*tx * cameras[i].R*K0.inv();
		for (int j = 0; j < N; j++)
		{
			double x0 = (j / vertices[i].num_y)*vertices[i].dw;
			double y0 = (j % vertices[i].num_y)*vertices[i].dh;
			Mat p0(1, 3, CV_64F);
			p0.at<double>(0, 0) = x0;
			p0.at<double>(0, 1) = y0;
			p0.at<double>(0, 2) = 1.0;
			Mat n = p0 * E;
			_triplets.emplace_back(num_equ, (i*N + j) * 2, n.at<double>(0, 0));
			_triplets.emplace_back(num_equ, (i*N + j) * 2 + 1, n.at<double>(0, 1));
			_b_vector.emplace_back(num_equ, -n.at<double>(0, 2));
			num_equ++;
		}
	}
}

void prepareGlobalSimilarityTerm(vector<Triplet<double> > & _triplets, vector<pair<int, double> > & _b_vector,
	const MatchingPointsandIndex &mpi, const vector<MatchInfo> &info,
	const vector<detail::CameraParams> &cameras, const vector<VerticesInfo> &vertices, int AlignEquNum, double w)
{
	int N = vertices[0].num_x*vertices[0].num_y;
	int num_equ = AlignEquNum;
	Mat K0 = Mat::eye(3, 3, CV_64F);
	K0.at<double>(0, 0) = cameras[0].focal;
	K0.at<double>(1, 1) = cameras[0].focal;
	K0.at<double>(0, 2) = cameras[0].ppx;
	K0.at<double>(1, 2) = cameras[0].ppy;
	for (int i = 0; i < vertices.size(); i++)
	{
		Mat K = Mat::eye(3, 3, CV_64F);
		K.at<double>(0, 0) = cameras[i].focal;
		K.at<double>(1, 1) = cameras[i].focal;
		K.at<double>(0, 2) = cameras[i].ppx;
		K.at<double>(1, 2) = cameras[i].ppy;
		for (int j = 0; j < N; j++)
		{
			double x0 = (j / vertices[i].num_y)*vertices[i].dw;
			double y0 = (j % vertices[i].num_y)*vertices[i].dh;
			Point2f p(x0, y0);
			vector<double> dis;
			pointsDistance(p, mpi.points[i], dis);

			vector<int> sort_idx;
			sortIdx(Mat(dis).t(), sort_idx, SORT_EVERY_ROW + SORT_ASCENDING);
						
			double sigma = 0.5*dis[sort_idx[5]];
			double gamma = 0;
			Mat n = Mat::zeros(1, 3, CV_64F);
			double wi_sum = 0.0;
			for (int k = 0; k < dis.size(); ++k)
			{
				//double wi = std::exp(-dis[k] * dis[k] / sigma / sigma);
				double wi = std::exp(-(dis[k] - dis[sort_idx[5]])*(dis[k] - dis[sort_idx[5]]) / sigma/sigma);
				n = n + wi*info[mpi.img_index[i][k]].class_normal[mpi.class_index[i][k]];
				wi_sum += wi;
			}
			n = n / wi_sum;

			Mat p0(3, 1, CV_64F);
			p0.at<double>(0, 0) = x0;
			p0.at<double>(1, 0) = y0;
			p0.at<double>(2, 0) = 1.0;
			Mat p_ = K0*(cameras[i].R+cameras[i].t*n).inv()*K.inv()*p0;
			_triplets.emplace_back(num_equ, (i*N + j) * 2, w);
			_b_vector.emplace_back(num_equ, w*p_.at<double>(0, 0) / p_.at<double>(2, 0));
			_triplets.emplace_back(num_equ + 1, (i*N + j) * 2 + 1, w);
			_b_vector.emplace_back(num_equ + 1, w*p_.at<double>(1, 0) / p_.at<double>(2, 0));
			num_equ += 2;
		}
	}
}

void prepareLocalSimilarityTerm(vector<Triplet<double> > & _triplets, const vector<VerticesInfo> &vertices, int LastEquNum, double w)
{
	int N = vertices[0].num_x*vertices[0].num_y;
	int num_equ = LastEquNum;
	for (int i = 0; i < vertices.size(); ++i)
	{
		for (int j = 0; j < N; j++)
		{
			if (j / vertices[i].num_y == 0 || j / vertices[i].num_y == (vertices[i].num_x - 1) || j % vertices[i].num_y == 0 || j % vertices[i].num_y == (vertices[i].num_y - 1))
				continue;
			_triplets.emplace_back(num_equ, (i*N + j) * 2, w);
			_triplets.emplace_back(num_equ, (i*N + j - 1) * 2, -0.25*w);
			_triplets.emplace_back(num_equ, (i*N + j + 1) * 2, -0.25*w);
			_triplets.emplace_back(num_equ, (i*N + j - vertices[i].num_y) * 2, -0.25*w);
			_triplets.emplace_back(num_equ, (i*N + j + vertices[i].num_y) * 2, -0.25*w);
			_triplets.emplace_back(num_equ + 1, (i*N + j) * 2 + 1, w);
			_triplets.emplace_back(num_equ + 1, (i*N + j - 1) * 2 + 1, -0.25*w);
			_triplets.emplace_back(num_equ + 1, (i*N + j + 1) * 2 + 1, -0.25*w);
			_triplets.emplace_back(num_equ + 1, (i*N + j - vertices[i].num_y) * 2 + 1, -0.25*w);
			_triplets.emplace_back(num_equ + 1, (i*N + j + vertices[i].num_y) * 2 + 1, -0.25*w);
			num_equ += 2;
		}
	}
}

void gatherMatchingPoints(const vector<MatchInfo> &info, MatchingPointsandIndex & result, const int image_num)
{
	result.points.resize(image_num);
	result.class_index.resize(image_num);
	result.img_index.resize(image_num);

	for (int i = 0; i < info.size(); i++)
	{
		result.points[info[i].img_idx1].insert(result.points[info[i].img_idx1].end(), info[i].re_points1.begin(), info[i].re_points1.end());
		result.img_index[info[i].img_idx1].insert(result.img_index[info[i].img_idx1].end(), info[i].class_idx.size(), i);
		result.class_index[info[i].img_idx1].insert(result.class_index[info[i].img_idx1].end(), info[i].class_idx.begin(), info[i].class_idx.end());
		result.points[info[i].img_idx2].insert(result.points[info[i].img_idx2].end(), info[i].re_points2.begin(), info[i].re_points2.end());
		result.img_index[info[i].img_idx2].insert(result.img_index[info[i].img_idx2].end(), info[i].class_idx.size(), i);
		result.class_index[info[i].img_idx2].insert(result.class_index[info[i].img_idx2].end(), info[i].class_idx.begin(), info[i].class_idx.end());
	}
}

void calcBorders(const vector<vector<Point2f>> &_proj_vertices, 
	float &max_x, float &max_y, float &min_x, float &min_y)
{
	max_x = -10000.0;
	max_y = -10000.0;
	min_x = 10000.0;
	min_y = 10000.0;
	for (int i = 0; i < _proj_vertices.size(); i++)
	{
		for (int j = 0; j < _proj_vertices[i].size(); j++)
		{
			if (_proj_vertices[i][j].x > max_x)
				max_x = _proj_vertices[i][j].x;
			if (_proj_vertices[i][j].x < min_x)
				min_x = _proj_vertices[i][j].x;
			if (_proj_vertices[i][j].y > max_y)
				max_y = _proj_vertices[i][j].y;
			if (_proj_vertices[i][j].y < min_y)
				min_y = _proj_vertices[i][j].y;
		}
	}
}

void image_warping(const Mat &image, Mat &warped_image, const VerticesInfo & _vertices, const vector<Point2f> &_proj_vertices, float &min_x, float &min_y)
{
	for (int i = 0; i < _vertices.num_x - 1; i++)
	{
		for (int j = 0; j < _vertices.num_y - 1; j++)
		{
			vector<Point2f> v, proj_v;
			v.push_back(Point2f(i*_vertices.dw, j*_vertices.dh));
			v.push_back(Point2f(i*_vertices.dw, (j + 1)*_vertices.dh));
			v.push_back(Point2f((i + 1)*_vertices.dw, (j + 1)*_vertices.dh));
			v.push_back(Point2f((i + 1)*_vertices.dw, j*_vertices.dh));
			proj_v.push_back(_proj_vertices[i*_vertices.num_y + j]);
			proj_v.push_back(_proj_vertices[i*_vertices.num_y + j + 1]);
			proj_v.push_back(_proj_vertices[(i + 1)*_vertices.num_y + j + 1]);
			proj_v.push_back(_proj_vertices[(i + 1)*_vertices.num_y + j]);
			Mat H = findHomography(v, proj_v);

			double minVal_x=10000.0, maxVal_x=-10000.0, minVal_y=10000.0, maxVal_y=-10000.0;
			for (int k = 0; k < proj_v.size(); k++)
			{
				if (proj_v[k].x > maxVal_x)
					maxVal_x = proj_v[k].x;
				if (proj_v[k].y > maxVal_y)
					maxVal_y = proj_v[k].y;
				if (proj_v[k].x < minVal_x)
					minVal_x = proj_v[k].x;
				if (proj_v[k].y < minVal_y)
					minVal_y = proj_v[k].y;
			}
			for (int x = cvFloor(minVal_x - min_x); x <= cvCeil(maxVal_x - min_x); x++)
			{
				for (int y = cvFloor(minVal_y - min_y); y <= cvCeil(maxVal_y - min_y); y++)
				{
					if (pointPolygonTest(proj_v, Point2f(x + min_x, y + min_y), false) == 1)
					{
						Point2f pt = applyTransform3x3(x + min_x, y + min_y, H.inv());
						if (pt.x >= 0 && pt.x < image.cols&&pt.y >= 0 && pt.y < image.rows)
							warped_image.at<Vec3b>(Point(x, y)) = image.at<Vec3b>(Point((int)pt.x, (int)pt.y));
						else
							warped_image.at<Vec3b>(Point(x, y)) = Vec3b(0, 0, 0);
					}
				}
			}
		}
	}
}

void rectifyImage(const Mat &src, Mat &dst, const CameraParams &camera)
{
	Mat K = Mat::eye(3, 3, CV_64F);
	K.at<double>(0, 0) = camera.focal;
	K.at<double>(1, 1) = camera.focal;
	K.at<double>(0, 2) = camera.ppx;
	K.at<double>(1, 2) = camera.ppy;

	Mat H = K * camera.R.inv()*K.inv();
	vector<vector<Point2f>> proj_vertices(1);
	proj_vertices[0].push_back(applyTransform3x3(0.0, 0.0, H));
	proj_vertices[0].push_back(applyTransform3x3(double(src.cols), 0.0, H));
	proj_vertices[0].push_back(applyTransform3x3(double(src.cols), double(src.rows), H));
	proj_vertices[0].push_back(applyTransform3x3(0.0, double(src.rows), H));

	float max_x, max_y, min_x, min_y;
	calcBorders(proj_vertices, max_x, max_y, min_x, min_y);
	float cw = max_x - min_x;
	float ch = max_y - min_y;
	
	image_warping(src, dst, H, cw, ch, min_x, min_y);
}

double alignmentSSIM(const vector<Mat>& images)
{
	vector<Mat> grayimgs;
	for (int n = 0; n < images.size(); n++)
	{
		Mat gray_img;
		cvtColor(images[n], gray_img, CV_BGR2GRAY);
		grayimgs.push_back(gray_img);
	}

	double ux = 0.0, uy = 0.0;
	double sigmax = 0.0, sigmay = 0.0, sigmaxy = 0.0;;
	int N = 0;
	for (int n = 0; n < images.size() - 1; n++)
	{
		for (int m = n+1; m < images.size(); m++)
		{
			for (int i = 0; i < grayimgs[n].rows; i++)
			{
				const uchar* pdata1 = grayimgs[n].ptr<uchar>(i);
				const uchar* pdata2 = grayimgs[m].ptr<uchar>(i);
				//Vec3b* presult = result.ptr<Vec3b>(i);
				for (int j = 0; j < grayimgs[n].cols; j++)
				{
					bool b1 = true, b2 = true;
					if (pdata1[j] == 0)
						b1 = false;
					if (pdata2[j] == 0)
						b2 = false;
					if (b1&&b2)
					{
						ux += (double)pdata1[j];
						uy += (double)pdata2[j];
						sigmax += ((double)pdata1[j])*((double)pdata1[j]);
						sigmay += ((double)pdata2[j])*((double)pdata2[j]);
						sigmaxy += ((double)pdata1[j])*((double)pdata2[j]);
						N++;
					}

				}
			}
		}
	}

	ux = ux / N;
	uy = uy / N;
	sigmax = sqrt((sigmax - N * ux*ux) / (N - 1));
	sigmay = sqrt((sigmay - N * uy*uy) / (N - 1));
	sigmaxy = (sigmaxy - N * ux*uy) / (N - 1);
	double L = 2 * ux*uy / (ux*ux + uy * uy);
	double C = 2 * sigmax*sigmay / (sigmax*sigmax + sigmay * sigmay);
	double S = sigmaxy / (sigmax*sigmay);
	double SSIM = L * C*S;
	return SSIM;
}

int main()
{
	vector<string> img_names;
	string file_name = "REW_worktable";
	string file_dir = "./input-42-data/" + file_name + "/";
	img_names = getImageFileFullNamesInDir(file_dir);
	int num_images = static_cast<int>(img_names.size());
	vector<Mat> full_imgs;

	vector<vector<Point2f>> all_imgs_keypoints(num_images);
	vector<Mat> all_imgs_feature_descriptors(num_images);
	for (int i = 0; i < num_images; ++i)
	{
		Mat full_img = imread(img_names[i]);
		//resize(full_img, full_img, Size(full_img.cols / 2, full_img.rows / 2));
		full_imgs.push_back(full_img);
		Mat gray_img;
		cvtColor(full_img, gray_img, CV_BGR2GRAY);

		vector<Point2f> key_points;
		Mat feature_descriptors;
		featureDetect(gray_img, key_points, feature_descriptors);
		all_imgs_keypoints[i] = key_points;
		all_imgs_feature_descriptors[i] = feature_descriptors;

#ifdef IMAGEDEBUG
		vector<Point2f> feature_points = key_points;
		Mat feature_img;
		full_img.copyTo(feature_img);
		for (int j = 0; j < feature_points.size(); j++)
		{
			circle(feature_img, cv::Point((int)feature_points[j].x, (int)feature_points[j].y), 3, cv::Scalar(0, 255, 0));
		}
#endif // IMAGEDEBUG
	}
#ifdef IMAGEDEBUG
	int combine_rows = (std::max)(full_imgs[0].rows, full_imgs[1].rows);
	Mat match_img(combine_rows, full_imgs[0].cols + full_imgs[1].cols, full_imgs[0].type());
#endif	
	vector<MatchInfo> all_matchinfo;
	for (int i = 0; i < num_images; ++i)
	{
		for (int j = i + 1; j < num_images; ++j)
		{
			vector<int> match1, match2;
			featureMatch(all_imgs_feature_descriptors[i], all_imgs_feature_descriptors[j], match1, match2);
			vector<Point2f> points1, points2;
			for (int k = 0; k < match1.size(); k++)
			{
				points1.push_back(all_imgs_keypoints[i][match1[k]]);
				points2.push_back(all_imgs_keypoints[j][match2[k]]);
			}
#ifdef IMAGEDEBUG
			full_imgs[i].copyTo(match_img(Range(0, full_imgs[0].rows), Range(0, full_imgs[0].cols)));
			full_imgs[j].copyTo(match_img(Range(0, full_imgs[1].rows), Range(full_imgs[0].cols, full_imgs[0].cols + full_imgs[1].cols)));
			for (int i = 0; i < points1.size(); i++)
			{
				circle(match_img, cv::Point(int(points1[i].x), int(points1[i].y)), 5, cv::Scalar(255, 0, 0), 2);
				circle(match_img, cv::Point(int(points2[i].x) + full_imgs[0].cols, int(points2[i].y)), 5, cv::Scalar(255, 0, 0), 2);
				//line(match_img, cv::Point(int(points1[i].x), int(points1[i].y)), cv::Point(int(points2[i].x) + full_imgs[0].cols, int(points2[i].y)), cv::Scalar(255, 0, 0), 1);
			}
#endif // IMAGEDEBUG
			MatchInfo match_info;
			pointsRansac(points1, points2, match1, match2, match_info);
#ifdef IMAGEDEBUG
			for (int i = 0; i < match_info.re_points1.size(); i++)
			{
				circle(match_img, cv::Point(int(match_info.re_points1[i].x), int(match_info.re_points1[i].y)), 5, cv::Scalar(0, 255, 0), 2);
				circle(match_img, cv::Point(int(match_info.re_points2[i].x) + full_imgs[0].cols, int(match_info.re_points2[i].y)), 5, cv::Scalar(0, 255, 0), 2);
				//line(match_img, cv::Point(int(match_info.re_points1[i].x), int(match_info.re_points1[i].y)), cv::Point(int(match_info.re_points2[i].x) + full_imgs[0].cols, int(match_info.re_points2[i].y)), cv::Scalar(0, 255, 0), 1);
			}
#endif // IMAGEDEBUG
			match_info.img_idx1 = i;
			match_info.img_idx2 = j;
			all_matchinfo.push_back(match_info);
		}
	}
	MatchingPointsandIndex mpi;
	gatherMatchingPoints(all_matchinfo, mpi, num_images);

	vector<Vec2i> stitch_order;
	int center_image_idx = findStitchingOrder(num_images, all_matchinfo, stitch_order);
	cout << "center_image_idx: " << center_image_idx << endl;
	double focal = estimateFocal(full_imgs, all_matchinfo);
	cout << "focal: " << focal << endl;
	calcEssentialMat(full_imgs, all_matchinfo, stitch_order, focal);
	vector<detail::CameraParams> cameras(full_imgs.size());
	estimateCameras(full_imgs, all_matchinfo, stitch_order, focal, cameras);
	
	compromiseCameras(cameras);
	//for (int i = 0; i < full_imgs.size(); i++)
	//{
	//	Mat rectified_img;
	//	rectifyImage(full_imgs[i], rectified_img, cameras[i]);
	//}
	
	calcNormalVector(all_matchinfo, cameras);
	BundleAdjustment ba;
	ba.estimate(all_imgs_keypoints, all_matchinfo, cameras);

	vector<VerticesInfo> vertices(full_imgs.size());
	for (int i = 0; i < full_imgs.size(); i++)
	{
		vertices[i].num_x = full_imgs[i].cols / 40 + 1;
		vertices[i].dw = (float)(full_imgs[i].cols) / vertices[i].num_x;
		vertices[i].num_x++;
		vertices[i].num_y = full_imgs[i].rows / 40 + 1;
		vertices[i].dh = (float)(full_imgs[i].rows) / vertices[i].num_y;
		vertices[i].num_y++;
	}

	vector<Triplet<double> > triplets;
	vector<pair<int, double> > b_vector;
	int AlignEquNum = 0, GlobalSimilarEquNum = 0, LocalSimilarEquNum = 0;
	for (int i = 0; i < all_matchinfo.size(); i++)
	{
		AlignEquNum += 2 * all_matchinfo[i].re_points1.size();
	}
	GlobalSimilarEquNum = vertices[0].num_x*vertices[0].num_y*vertices.size()*2;
	LocalSimilarEquNum = (vertices[0].num_x - 2)*(vertices[0].num_y - 2)*vertices.size() * 2;
	//triplets.reserve(AlignEquNum + SimilarEquNum);
	//b_vector.reserve(AlignEquNum + SimilarEquNum);

	prepareAlignmentTerm(triplets, vertices, all_matchinfo);
	//prepareSimilarityTerm(triplets, b_vector, cameras, vertices, AlignEquNum);
	double global_similarity_weight = 1;
	double local_similarity_weight = 0.1;
	prepareGlobalSimilarityTerm(triplets, b_vector, mpi, all_matchinfo, cameras, vertices, AlignEquNum, global_similarity_weight);
	prepareLocalSimilarityTerm(triplets, vertices, AlignEquNum + GlobalSimilarEquNum, local_similarity_weight);

	vector<vector<Point2f>> proj_vertices(vertices.size());
	LeastSquaresSolve(triplets, b_vector, proj_vertices, AlignEquNum + GlobalSimilarEquNum + LocalSimilarEquNum, GlobalSimilarEquNum/2);

	float max_x, max_y, min_x, min_y;
	calcBorders(proj_vertices, max_x, max_y, min_x, min_y);

	vector<Mat> warped_imgs;
	for (int i = 0; i < full_imgs.size(); i++)
	{
		Mat warped_img = Mat::zeros(cvCeil(max_y - min_y), cvCeil(max_x - min_x), full_imgs[i].type());
		image_warping(full_imgs[i], warped_img, vertices[i], proj_vertices[i], min_x, min_y);
		warped_imgs.push_back(warped_img);
	}
	Mat result;
	image_blending(warped_imgs, result);
	string result_name = "result/" + file_name + ".jpg";
	imwrite(result_name, result);
	double SSIM = alignmentSSIM(warped_imgs);
	std::cout << "SSIM= " << SSIM << endl;
	cout << "end";
}