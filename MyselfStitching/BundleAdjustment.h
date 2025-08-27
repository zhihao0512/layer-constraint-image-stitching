#pragma once
#ifndef _BUNDLE_ADJUSTMENT_
#define _BUNDLE_ADJUSTMENT_

#include <opencv2\core\core.hpp>
#include <opencv2\core\types.hpp>
#include <opencv2\stitching\detail\camera.hpp>

//using namespace cv;
//using namespace cv::detail;

struct MatchInfo
{
	bool isUse;
	int img_idx1, img_idx2;
	cv::Mat H;
	cv::Mat R, t;
	std::vector<cv::Mat> class_normal;
	std::vector<cv::Mat> class_H;
	std::vector<int> class_idx;
	std::vector<int> main_class_idx;
	std::vector<int> inliers_match1, inliers_match2;
	std::vector<cv::Point2f> re_points1, re_points2;
	std::vector<cv::Point2f> main_points1, main_points2;
	int class_num;
};

class BundleAdjustment
{
public:

	cv::TermCriteria termCriteria() { return term_criteria_; }
	void setTermCriteria(const cv::TermCriteria& term_criteria) { term_criteria_ = term_criteria; }

	BundleAdjustment()
		: num_images_(0), total_num_matches_(0), num_planes_(0),
		num_params_per_cam_(7),
		num_errs_per_measurement_(3),
		all_keypoints_(0), info_(0)
	{
		setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 1e-3));
	}

	// Runs bundle adjustment
	virtual bool estimate(const std::vector<std::vector<cv::Point2f>> &all_keypoints, 
		std::vector<MatchInfo> &info,
		std::vector<cv::detail::CameraParams> &cameras);

private:

	/** @brief Sets initial camera parameter to refine.

	@param cameras Camera parameters
	*/
	void setUpInitialCameraParams(const std::vector<cv::detail::CameraParams> &cameras, const vector<MatchInfo> &info);
	/** @brief Gets the refined camera parameters.

	@param cameras Refined camera parameters
	*/
	void obtainRefinedCameraParams(std::vector<cv::detail::CameraParams> &cameras, vector<MatchInfo> &info);
	/** @brief Calculates error vector.

	@param err Error column-vector of length total_num_matches \* num_errs_per_measurement
	*/
	void calcError(cv::Mat &err);
	/** @brief Calculates the cost function jacobian.

	@param jac Jacobian matrix of dimensions
	(total_num_matches \* num_errs_per_measurement) x (num_images \* num_params_per_cam)
	*/
	void calcJacobian(cv::Mat &jac);

	// 3x3 8U mask, where 0 means don't refine respective parameter, != 0 means refine
	cv::Mat refinement_mask_;

	int num_images_;
	int total_num_matches_;
	int num_planes_;

	int num_params_per_cam_;
	int num_errs_per_measurement_;

	const std::vector<cv::Point2f> *all_keypoints_;
	const MatchInfo *info_;
	cv::detail::CameraParams *cameras_;

	//Levenberg-Marquardt algorithm termination criteria
	cv::TermCriteria term_criteria_;

	// Camera parameters matrix (CV_64F)
	cv::Mat cam_params_;
	cv::Mat plane_params_;
	cv::Mat all_params_;

	// Connected images pairs
	std::vector<std::pair<int, int> > edges_;
};

#endif // !_BUNDLE_ADJUSTMENT_

