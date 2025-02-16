#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Essential matrix estimator from corresponding normalized point pairs.
//
// This algorithm solves the 5-Point problem based on the following paper:
//
//    D. Nister, An efficient solution to the five-point relative pose problem,
//    IEEE-T-PAMI, 26(6), 2004.
//    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.86.8769
class EssentialMatrixFivePointEstimator {
 public:
  typedef Eigen::Vector3d X_t;
  typedef Eigen::Vector3d Y_t;
  typedef Eigen::Matrix3d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 5;

  // Estimate up to 10 possible essential matrix solutions from a set of
  // corresponding points.
  //
  //  The number of corresponding points must be at least 5.
  //
  // @param points1  First set of corresponding points.
  // @param points2  Second set of corresponding points.
  //
  // @return         Up to 10 solutions as a vector of 3x3 essential matrices.
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // Calculate the residuals of a set of corresponding points and a given
  // essential matrix.
  //
  // Residuals are defined as the squared Sampson error.
  //
  // @param points1    First set of corresponding points.
  // @param points2    Second set of corresponding points.
  // @param E          3x3 essential matrix.
  // @param residuals  Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& E,
                        std::vector<double>* residuals);
};

}  // namespace colmap
