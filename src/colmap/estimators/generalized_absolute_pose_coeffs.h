#pragma once

#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>

namespace colmap {

Eigen::Matrix<double, 9, 1> ComputeDepthsSylvesterCoeffs(
    const Eigen::Matrix<double, 3, 6>& K);

}  // namespace colmap
