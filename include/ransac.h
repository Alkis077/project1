#ifndef RANSAC_H
#define RANSAC_H

#include <random>
#include <vector>

#include <Eigen/Dense>

#include "matching.h"

struct RansacParams {
    double tau = 3;        // inlier threshold on the reprojection error (pixels)
    int Tmax = 2000;       // maximum number of trials
    double p = 0.99;       // wanted probability of drawing one sample of 4 inliers
    bool adaptive = true;  // lower the trial limit from the inlier ratio
};

struct RansacResult {
    bool ok = false;           // false: fewer than 4 matches or no valid homography
    Eigen::Matrix3d H;         // final homography (after the refit)
    std::vector<int> inliers;  // final inliers among all matches
    double medianError = 0;    // median reprojection error of the final inliers
    int trials = 0;            // trials that were run
};

// RANSAC with samples of 4 matches. Every sample that is not degenerate gives
// a homography with the normalized DLT, and the one with the most inliers is
// kept (ties: smaller median error of its inliers). Then the homography is
// estimated again from those inliers only (refit) and the final inliers are
// found again among all matches.
RansacResult ransac(const std::vector<Match> &matches, const RansacParams &params,
                    std::mt19937 &rng);

// true if a set of matches is degenerate in one of the two images: fewer than
// 4 different points, or all of them (almost) on one line. Used for the final
// inliers: e.g. many points of the query matched to the same point of the
// candidate give a homography that maps a whole region to one point.
bool degenerateInliers(const std::vector<Match> &matches, const std::vector<int> &idx);

#endif
