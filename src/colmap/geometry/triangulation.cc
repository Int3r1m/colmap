// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/geometry/triangulation.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Dense>

namespace colmap {

bool TriangulatePoint(const Eigen::Matrix3x4d& cam1_from_world,
                      const Eigen::Matrix3x4d& cam2_from_world,
                      const Eigen::Vector3d& point1,
                      const Eigen::Vector3d& point2,
                      Eigen::Vector3d* xyz) {
  THROW_CHECK_NOTNULL(xyz);

  Eigen::Matrix<double, 6, 4> A;
  A.row(0) =
      -point1(2) * cam1_from_world.row(1) + point1(1) * cam1_from_world.row(2);
  A.row(1) =
      point1(2) * cam1_from_world.row(0) - point1(0) * cam1_from_world.row(2);
  A.row(2) =
      -point1(1) * cam1_from_world.row(0) + point1(0) * cam1_from_world.row(1);
  A.row(3) =
      -point2(2) * cam2_from_world.row(1) + point2(1) * cam2_from_world.row(2);
  A.row(4) =
      point2(2) * cam2_from_world.row(0) - point2(0) * cam2_from_world.row(2);
  A.row(5) =
      -point2(1) * cam2_from_world.row(0) + point2(0) * cam2_from_world.row(1);

  const Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd(A,
                                                          Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
  if (svd.info() != Eigen::Success) {
    return false;
  }
#endif

  if (svd.matrixV()(3, 3) == 0) {
    return false;
  }

  *xyz = svd.matrixV().col(3).hnormalized();
  return true;
}

bool TriangulateMultiViewPoint(
    const std::vector<Eigen::Matrix3x4d>& cams_from_world,
    const std::vector<Eigen::Vector3d>& points,
    Eigen::Vector3d* xyz) {
  THROW_CHECK_EQ(cams_from_world.size(), points.size());
  THROW_CHECK_NOTNULL(xyz);

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  for (size_t i = 0; i < points.size(); i++) {
    const Eigen::Vector3d point = points[i].normalized();
    const Eigen::Matrix3x4d term =
        cams_from_world[i] - point * point.transpose() * cams_from_world[i];
    A += term.transpose() * term;
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);
  if (eigen_solver.info() != Eigen::Success ||
      eigen_solver.eigenvectors()(3, 0) == 0) {
    return false;
  }

  *xyz = eigen_solver.eigenvectors().col(0).hnormalized();
  return true;
}

bool TriangulateIDWMidpoint(const Eigen::Matrix3x4d& cam1_from_world,
                            const Eigen::Matrix3x4d& cam2_from_world,
                            const Eigen::Vector3d& point1,
                            const Eigen::Vector3d& point2,
                            Eigen::Vector3d& point3D) {
  const Eigen::Matrix3d R0 = cam1_from_world.leftCols<3>();
  const Eigen::Vector3d t0 = cam1_from_world.col(3);
  const Eigen::Matrix3d R1 = cam2_from_world.leftCols<3>();
  const Eigen::Vector3d t1 = cam2_from_world.col(3);
  const Eigen::Matrix3d R = R1 * R0.transpose();
  const Eigen::Vector3d t = t1 - R * t0;
  const Eigen::Vector3d Rx0 = R * point1;

  const double p_norm = Rx0.cross(point2).norm();
  const double q_norm = Rx0.cross(t).norm();
  const double r_norm = point2.cross(t).norm();

  const auto xprime1 =
      (q_norm / (q_norm + r_norm)) * (t + (r_norm / p_norm) * (Rx0 + point2));

  point3D = R1.transpose() * (xprime1 - t1);

  const Eigen::Vector3d lambda0_Rx0 = (r_norm / p_norm) * Rx0;
  const Eigen::Vector3d lambda1_x1 = (q_norm / p_norm) * point2;

  return (t + lambda0_Rx0 - lambda1_x1).squaredNorm() <
         std::min(std::min((t + lambda0_Rx0 + lambda1_x1).squaredNorm(),
                           (t - lambda0_Rx0 - lambda1_x1).squaredNorm()),
                  (t - lambda0_Rx0 + lambda1_x1).squaredNorm());
}

bool TriangulateMultiViewPointIGG(
    const std::vector<Eigen::Matrix3x4d>& cams_from_world,
    const std::vector<Eigen::Vector3d>& points,
    std::vector<double>& weights,
    Eigen::Vector3d& point3D) {
  // Triangulate the point3D
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3d point = points[i].normalized();
    const Eigen::Matrix3x4d term =
        cams_from_world[i] - point * point.transpose() * cams_from_world[i];
    A += weights[i] * term.transpose() * term;
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);
  if (eigen_solver.info() != Eigen::Success ||
      eigen_solver.eigenvectors()(3, 0) == 0) {
    return false;
  }

  const Eigen::Vector3d point3D_previous = point3D;
  point3D = eigen_solver.eigenvectors().col(0).hnormalized();

  // Check if the point3D has converged
  if ((point3D - point3D_previous).norm() < 1e-2) {
    return true;
  }

  // Update the weights
  double sum = 0;
  std::vector<double> distances(points.size(), 0.0);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector4d point3D_homogeneous = point3D.homogeneous();
    const Eigen::Vector3d point = points[i].normalized();
    const Eigen::Matrix3x4d term =
        cams_from_world[i] - point * point.transpose() * cams_from_world[i];
    distances[i] = (term * point3D_homogeneous).norm();
    sum += weights[i] * distances[i] * distances[i];
  }

  const double k0 = 1.5;
  const double k1 = 2.5;
  const double sigma = std::sqrt(sum / (points.size() - 2.0));
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double u = std::abs(distances[i] / sigma);
    if (u < k0) {
      weights[i] = 1.0;
    } else if (u >= k0 && u < k1) {
      weights[i] = k0 / u;
    } else {
      weights[i] = 0.0;
    }
  }

  // Continue iteration
  return TriangulateMultiViewPointIGG(
      cams_from_world, points, weights, point3D);
}

bool TriangulateOptimalPoint(const Eigen::Matrix3x4d& cam1_from_world_mat,
                             const Eigen::Matrix3x4d& cam2_from_world_mat,
                             const Eigen::Vector3d& point1,
                             const Eigen::Vector3d& point2,
                             Eigen::Vector3d* xyz) {
  const Rigid3d cam1_from_world(
      Eigen::Quaterniond(cam1_from_world_mat.leftCols<3>()),
      cam1_from_world_mat.col(3));
  const Rigid3d cam2_from_world(
      Eigen::Quaterniond(cam2_from_world_mat.leftCols<3>()),
      cam2_from_world_mat.col(3));
  const Rigid3d cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world);
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  Eigen::Vector3d optimal_point1;
  Eigen::Vector3d optimal_point2;
  FindOptimalImageObservations(
      E, point1, point2, &optimal_point1, &optimal_point2);

  return TriangulatePoint(cam1_from_world_mat,
                          cam2_from_world_mat,
                          optimal_point1,
                          optimal_point2,
                          xyz);
}

double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3D) {
  const double baseline_length_squared =
      (proj_center1 - proj_center2).squaredNorm();

  const double ray_length_squared1 = (point3D - proj_center1).squaredNorm();
  const double ray_length_squared2 = (point3D - proj_center2).squaredNorm();

  // Using "law of cosines" to compute the enclosing angle between rays.
  const double denominator =
      2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
  if (denominator == 0.0) {
    return 0.0;
  }
  const double nominator =
      ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
  const double angle =
      std::abs(std::acos(std::clamp(nominator / denominator, -1.0, 1.0)));

  // Triangulation is unstable for acute angles (far away points) and
  // obtuse angles (close points), so always compute the minimum angle
  // between the two intersecting rays.
  return std::min(angle, M_PI - angle);
}

std::vector<double> CalculateTriangulationAngles(
    const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const std::vector<Eigen::Vector3d>& points3D) {
  // Baseline length between camera centers.
  const double baseline_length_squared =
      (proj_center1 - proj_center2).squaredNorm();

  std::vector<double> angles(points3D.size());

  for (size_t i = 0; i < points3D.size(); ++i) {
    // Ray lengths from cameras to point.
    const double ray_length_squared1 =
        (points3D[i] - proj_center1).squaredNorm();
    const double ray_length_squared2 =
        (points3D[i] - proj_center2).squaredNorm();

    // Using "law of cosines" to compute the enclosing angle between rays.
    const double denominator =
        2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
    if (denominator == 0.0) {
      angles[i] = 0.0;
      continue;
    }
    const double nominator =
        ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
    const double angle = std::abs(std::acos(nominator / denominator));

    // Triangulation is unstable for acute angles (far away points) and
    // obtuse angles (close points), so always compute the minimum angle
    // between the two intersecting rays.
    angles[i] = std::min(angle, M_PI - angle);
  }

  return angles;
}

}  // namespace colmap
