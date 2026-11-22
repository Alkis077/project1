#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H

#include <vector>

#include <Eigen/Dense>

#include "matching.h"

// Thresholds of the checks (values suggested in the course notes for double)
const double EPS_COL = 1e-4;    // collinear triple: eta < EPS_COL
const double EPS_RANK = 1e-12;  // H is not invertible: sigma3 / sigma1 <= EPS_RANK
const double EPS_PROJ = 1e-12;  // projection: |c| <= EPS_PROJ * ||H row 3|| * ||p||

// Homography from the matches with the given indices (at least 4), with the
// normalized DLT: both point sets are moved to their centroid and scaled to
// mean distance sqrt(2), H_n is the right singular vector of the smallest
// singular value of A, and H = T'^-1 H_n T.
// Returns false if the points cannot be normalized or H is not valid.
bool estimateHomography(const std::vector<Match> &matches, const std::vector<int> &idx,
                        Eigen::Matrix3d &H);

// True if 4 matches cannot give a homography: two equal points or three
// collinear points in either image.
bool degenerateSample(const std::vector<Match> &matches, const int idx[4]);

// True if all elements of H are finite and H is numerically invertible.
bool validHomography(const Eigen::Matrix3d &H);

// Maps (x, y) with H: (a, b, c) = H (x, y, 1), result (a / c, b / c).
// Returns false if c is too close to 0 or the result is not finite.
bool project(const Eigen::Matrix3d &H, double x, double y, double &u, double &v);

// Reprojection error of a match: distance between (u, v) and the projection
// of (x, y). Infinity if the projection is not possible.
double reprojectionError(const Eigen::Matrix3d &H, const Match &m);

#endif
