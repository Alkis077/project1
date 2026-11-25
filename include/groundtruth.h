#ifndef GROUNDTRUTH_H
#define GROUNDTRUTH_H

#include <string>

#include <Eigen/Dense>

#include "dataset.h"

const double ACCURATE_PX = 5;   // successful estimate: corner error below this

// Reference homography H_1_j of HPatches (image 1 -> image j of the sequence).
// It is used only for the evaluation. Returns false if the file cannot be read.
bool readGroundTruth(const std::string &hpDir, const std::string &seq, int j, Eigen::Matrix3d &H);

// number j of an HPatches image ("v_bird/3" -> 3)
int imageNumber(const Image &img);

// width and height of an image file (false if it cannot be read)
bool imageSize(const std::string &path, int &width, int &height);

// Corner error of an estimated homography H against the reference Hgt:
// the four corners a_i of the first image are projected with both, and the
// error is the mean distance (pixels of the second image)
//     E = 1/4 * sum ||pi(H a_i) - pi(Hgt a_i)||
// Returns false if some corner cannot be projected (then there is no error).
bool cornerError(const Eigen::Matrix3d &H, const Eigen::Matrix3d &Hgt, int width, int height,
                 double &error);

#endif
