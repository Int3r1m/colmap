// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/math/polynomial.h"

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Eigenvalues>

namespace colmap {
namespace {

// Remove leading zero coefficients.
Eigen::VectorXd RemoveLeadingZeros(const Eigen::VectorXd& coeffs) {
  Eigen::VectorXd::Index num_zeros = 0;
  for (; num_zeros < coeffs.size(); ++num_zeros) {
    if (coeffs(num_zeros) != 0) {
      break;
    }
  }
  return coeffs.tail(coeffs.size() - num_zeros);
}

// Remove trailing zero coefficients.
Eigen::VectorXd RemoveTrailingZeros(const Eigen::VectorXd& coeffs) {
  Eigen::VectorXd::Index num_zeros = 0;
  for (; num_zeros < coeffs.size(); ++num_zeros) {
    if (coeffs(coeffs.size() - 1 - num_zeros) != 0) {
      break;
    }
  }
  return coeffs.head(coeffs.size() - num_zeros);
}

}  // namespace

bool FindLinearPolynomialRoots(const Eigen::VectorXd& coeffs,
                               Eigen::VectorXd* real,
                               Eigen::VectorXd* imag) {
  THROW_CHECK_EQ(coeffs.size(), 2);

  if (coeffs(0) == 0) {
    return false;
  }

  if (real != nullptr) {
    real->resize(1);
    (*real)(0) = -coeffs(1) / coeffs(0);
  }

  if (imag != nullptr) {
    imag->resize(1);
    (*imag)(0) = 0;
  }

  return true;
}

bool FindQuadraticPolynomialRoots(const Eigen::VectorXd& coeffs,
                                  Eigen::VectorXd* real,
                                  Eigen::VectorXd* imag) {
  THROW_CHECK_EQ(coeffs.size(), 3);

  const double a = coeffs(0);
  if (a == 0) {
    return FindLinearPolynomialRoots(coeffs.tail(2), real, imag);
  }

  const double b = coeffs(1);
  const double c = coeffs(2);
  if (b == 0 && c == 0) {
    if (real != nullptr) {
      real->resize(1);
      (*real)(0) = 0;
    }
    if (imag != nullptr) {
      imag->resize(1);
      (*imag)(0) = 0;
    }
    return true;
  }

  const double d = b * b - 4 * a * c;

  if (d >= 0) {
    const double sqrt_d = std::sqrt(d);
    if (real != nullptr) {
      real->resize(2);
      if (b >= 0) {
        (*real)(0) = (-b - sqrt_d) / (2 * a);
        (*real)(1) = (2 * c) / (-b - sqrt_d);
      } else {
        (*real)(0) = (2 * c) / (-b + sqrt_d);
        (*real)(1) = (-b + sqrt_d) / (2 * a);
      }
    }
    if (imag != nullptr) {
      imag->resize(2);
      imag->setZero();
    }
  } else {
    if (real != nullptr) {
      real->resize(2);
      real->setConstant(-b / (2 * a));
    }
    if (imag != nullptr) {
      imag->resize(2);
      (*imag)(0) = std::sqrt(-d) / (2 * a);
      (*imag)(1) = -(*imag)(0);
    }
  }

  return true;
}

int FindCubicPolynomialRoots(double c2,
                             double c1,
                             double c0,
                             Eigen::Vector3d* real) {
  constexpr double k2PiOver3 = 2.09439510239319526263557236234192;
  constexpr double k4PiOver3 = 4.18879020478639052527114472468384;
  const double c2_over_3 = c2 / 3.0;
  const double a = c1 - c2 * c2_over_3;
  double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
  double c = b * b / 4.0 + a * a * a / 27.0;
  int num_roots = 0;
  if (c > 0) {
    c = std::sqrt(c);
    b *= -0.5;
    (*real)[0] = std::cbrt(b + c) + std::cbrt(b - c) - c2_over_3;
    num_roots = 1;
  } else {
    c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
    double d = 2.0 * std::sqrt(-a / 3.0);
    const double acos_over_3 = std::acos(c) / 3.0;
    (*real)[0] = d * std::cos(acos_over_3) - c2_over_3;
    (*real)[1] = d * std::cos(acos_over_3 - k2PiOver3) - c2_over_3;
    (*real)[2] = d * std::cos(acos_over_3 - k4PiOver3) - c2_over_3;
    num_roots = 3;
  }

  // Single Newton iteration.
  for (int i = 0; i < num_roots; ++i) {
    const double x = (*real)[i];
    const double x2 = x * x;
    const double x3 = x * x2;
    const double dx =
        -(x3 + c2 * x2 + c1 * x + c0) / (3 * x2 + 2 * c2 * x + c1);
    (*real)[i] += dx;
  }

  return num_roots;
}

bool FindPolynomialRootsDurandKerner(const Eigen::VectorXd& coeffs_all,
                                     Eigen::VectorXd* real,
                                     Eigen::VectorXd* imag) {
  THROW_CHECK_GE(coeffs_all.size(), 2);

  const Eigen::VectorXd coeffs = RemoveLeadingZeros(coeffs_all);

  const int degree = coeffs.size() - 1;

  if (degree <= 0) {
    return false;
  } else if (degree == 1) {
    return FindLinearPolynomialRoots(coeffs, real, imag);
  } else if (degree == 2) {
    return FindQuadraticPolynomialRoots(coeffs, real, imag);
  }

  // Initialize roots.
  Eigen::VectorXcd roots(degree);
  roots(degree - 1) = std::complex<double>(1, 0);
  for (int i = degree - 2; i >= 0; --i) {
    roots(i) = roots(i + 1) * std::complex<double>(1, 1);
  }

  // Iterative solver.
  const int kMaxNumIterations = 100;
  const double kMaxRootChange = 1e-10;
  for (int iter = 0; iter < kMaxNumIterations; ++iter) {
    double max_root_change = 0.0;
    for (int i = 0; i < degree; ++i) {
      const std::complex<double> root_i = roots(i);
      std::complex<double> numerator = coeffs[0];
      std::complex<double> denominator = coeffs[0];
      for (int j = 0; j < degree; ++j) {
        numerator = numerator * root_i + coeffs[j + 1];
        if (i != j) {
          denominator = denominator * (root_i - roots(j));
        }
      }
      const std::complex<double> root_i_change = numerator / denominator;
      roots(i) = root_i - root_i_change;
      max_root_change =
          std::max(max_root_change, std::abs(root_i_change.real()));
      max_root_change =
          std::max(max_root_change, std::abs(root_i_change.imag()));
    }

    // Break, if roots do not change anymore.
    if (max_root_change < kMaxRootChange) {
      break;
    }
  }

  if (real != nullptr) {
    real->resize(degree);
    *real = roots.real();
  }
  if (imag != nullptr) {
    imag->resize(degree);
    *imag = roots.imag();
  }

  return true;
}

bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs_all,
                                        Eigen::VectorXd* real,
                                        Eigen::VectorXd* imag) {
  THROW_CHECK_GE(coeffs_all.size(), 2);

  Eigen::VectorXd coeffs = RemoveLeadingZeros(coeffs_all);

  const int degree = coeffs.size() - 1;

  if (degree <= 0) {
    return false;
  } else if (degree == 1) {
    return FindLinearPolynomialRoots(coeffs, real, imag);
  } else if (degree == 2) {
    return FindQuadraticPolynomialRoots(coeffs, real, imag);
  }

  // Remove the coefficients where zero is a solution.
  coeffs = RemoveTrailingZeros(coeffs);

  // Check if only zero is a solution.
  if (coeffs.size() == 1) {
    if (real != nullptr) {
      real->resize(1);
      (*real)(0) = 0;
    }
    if (imag != nullptr) {
      imag->resize(1);
      (*imag)(0) = 0;
    }
    return true;
  }

  // Fill the companion matrix.
  Eigen::MatrixXd C(coeffs.size() - 1, coeffs.size() - 1);
  C.setZero();
  for (Eigen::MatrixXd::Index i = 1; i < C.rows(); ++i) {
    C(i, i - 1) = 1;
  }
  C.row(0) = -coeffs.tail(coeffs.size() - 1) / coeffs(0);

  // Solve for the roots of the polynomial.
  Eigen::EigenSolver<Eigen::MatrixXd> solver(C, false);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  // If there are trailing zeros, we must add zero as a solution.
  const int effective_degree =
      coeffs.size() - 1 < degree ? coeffs.size() : coeffs.size() - 1;

  if (real != nullptr) {
    real->resize(effective_degree);
    real->head(coeffs.size() - 1) = solver.eigenvalues().real();
    if (effective_degree > coeffs.size() - 1) {
      (*real)(real->size() - 1) = 0;
    }
  }
  if (imag != nullptr) {
    imag->resize(effective_degree);
    imag->head(coeffs.size() - 1) = solver.eigenvalues().imag();
    if (effective_degree > coeffs.size() - 1) {
      (*imag)(imag->size() - 1) = 0;
    }
  }

  return true;
}

}  // namespace colmap
