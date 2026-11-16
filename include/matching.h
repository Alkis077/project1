#ifndef MATCHING_H
#define MATCHING_H

#include <vector>

#include "descriptors.h"

// One accepted correspondence: keypoint q of the first image (the query)
// and keypoint i of the second image (the candidate), with their positions.
struct Match {
    int q, i;
    double x, y;   // position in the first image
    double u, v;   // position in the second image
};

// For every descriptor of a, finds the two closest descriptors of b
// (Euclidean distance) and keeps the closest one when d1 < rho * d2
// (Lowe's ratio test). If b has fewer than two descriptors the test cannot
// be applied and no match is returned.
std::vector<Match> matchSift(const Descriptors &a, const Descriptors &b, double rho);

#endif
