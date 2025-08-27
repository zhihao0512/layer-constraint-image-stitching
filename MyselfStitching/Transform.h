//
//  Transform.h
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#ifndef __UglyMan_Stitiching__Transform__
#define __UglyMan_Stitiching__Transform__

//#include "../Configure.h"
#include <opencv2/core/core.hpp>
#include <vector>

//using namespace cv;
using namespace std;

cv::Mat getConditionerFromPts(const vector<cv::Point2f> & pts);

cv::Mat getNormalize2DPts(const vector<cv::Point2f> & pts,
                      vector<cv::Point2f> & newpts);

template <typename T>
T normalizeAngle(T x);
                 
template <typename T>
cv::Point_<T> applyTransform3x3(T x, T y, const cv::Mat & matT);

template <typename T>
T PointsTransformDistance(cv::Point_<T> pt1, cv::Point_<T> pt2, const cv::Mat & matT);

template <typename T>
cv::Point_<T> applyTransform2x3(T x, T y, const cv::Mat & matT);

template <typename T>
cv::Size_<T> normalizeVertices(vector<vector<cv::Point_<T> > > & vertices);

template <typename T>
cv::Rect_<T> getVerticesRects(const vector<cv::Point_<T> > & vertices);

template <typename T>
vector<cv::Rect_<T> > getVerticesRects(const vector<vector<cv::Point_<T> > > & vertices);

template <typename T>
T getSubpix(const cv::Mat & img, const cv::Point2f & pt);

template <typename T, size_t n>
cv::Vec<T, n> getSubpix(const cv::Mat & img, const cv::Point2f & pt);

template <typename T>
cv::Vec<T, 3> getEulerYZXRadians(const cv::Mat_<T> & rot_matrix);

template <typename T>
cv::Mat_<T> calYZXRadiansMatrix(const cv::Vec<T, 3> &angle);

template <typename T>
cv::Vec<T, 3> getEulerXYZRadians(const cv::Mat_<T> & rot_matrix);

template <typename T>
cv::Mat_<T> calXYZRadiansMatrix(const cv::Vec<T, 3> &angle);

template <typename T>
bool isEdgeIntersection(const cv::Point_<T> & src_1, const cv::Point_<T> & dst_1,
                        const cv::Point_<T> & src_2, const cv::Point_<T> & dst_2,
                        double * scale_1 = NULL, double * scale_2 = NULL);
template <typename T>
bool isRotationInTheRange(const T rotation, const T min_rotation, const T max_rotation);

#endif /* defined(__UglyMan_Stitiching__Transform__) */
