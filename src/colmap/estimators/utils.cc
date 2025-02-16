#include "colmap/estimators/utils.h"

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Geometry>

namespace colmap {

void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d>& points,
                                   std::vector<Eigen::Vector2d>* normed_points,
                                   Eigen::Matrix3d* normed_from_orig) {
  THROW_CHECK_GT(points.size(), 0);

  // Calculate centroid.
  Eigen::Vector2d centroid(0, 0);
  for (const Eigen::Vector2d& point : points) {
    centroid += point;
  }
  centroid /= points.size();

  // Root mean square distance to centroid of all points.
  double rms_mean_dist = 0;
  for (const Eigen::Vector2d& point : points) {
    rms_mean_dist += (point - centroid).squaredNorm();
  }
  rms_mean_dist = std::sqrt(rms_mean_dist / points.size());

  // Compose normalization matrix.
  const double norm_factor = std::sqrt(2.0) / rms_mean_dist;
  *normed_from_orig << norm_factor, 0, -norm_factor * centroid(0), 0,
      norm_factor, -norm_factor * centroid(1), 0, 0, 1;

  // Apply normalization matrix.
  normed_points->resize(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    (*normed_points)[i] =
        (*normed_from_orig * points[i].homogeneous()).hnormalized();
  }
}

void ComputeSquaredSampsonError(const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  residuals->resize(points1.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    const Eigen::Vector3d Ep1 = E * points1[i].homogeneous();
    const Eigen::Vector3d p2tE = E.transpose() * points2[i].homogeneous();
    const double p2tEp1 = p2tE.dot(points1[i].homogeneous());
    const Eigen::Vector4d denom(p2tE.x(), p2tE.y(), Ep1.x(), Ep1.y());
    (*residuals)[i] = p2tEp1 * p2tEp1 / denom.squaredNorm();
  }
}

void ComputeSquaredSampsonError(const std::vector<Eigen::Vector3d>& points1,
                                const std::vector<Eigen::Vector3d>& points2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  residuals->resize(points1.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    const Eigen::Vector3d Ep1 = E * points1[i];
    const Eigen::Vector3d p2tE = E.transpose() * points2[i];
    const double p2tEp1 = p2tE.dot(points1[i]);
    (*residuals)[i] =
        p2tEp1 * p2tEp1 / (Ep1.squaredNorm() + p2tE.squaredNorm());
  }
}

void ComputeSquaredReprojectionError(
    const std::vector<Eigen::Vector3d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3x4d& cam_from_world,
    std::vector<double>* residuals) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  residuals->resize(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    const Eigen::Vector3d point3D_in_cam =
        cam_from_world * points3D[i].homogeneous();
    if (points2D[i].norm() < 1.0 + std::numeric_limits<double>::epsilon()) {
      (*residuals)[i] = std::pow(
          std::acos(point3D_in_cam.normalized().dot(points2D[i])), 2.0);
    } else if (point3D_in_cam.z() < std::numeric_limits<double>::epsilon()) {
      (*residuals)[i] = std::numeric_limits<double>::max();
    } else {
      (*residuals)[i] =
          (point3D_in_cam.hnormalized().homogeneous() - points2D[i])
              .squaredNorm();
    }
  }
}

}  // namespace colmap
