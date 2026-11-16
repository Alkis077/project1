#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include <string>
#include <vector>

#include "dataset.h"

const int DIM = 128;   // length of a SIFT descriptor

// SIFT keypoints and descriptors of one image, stored row after row:
// descriptor i is data[i*DIM] ... data[i*DIM + DIM-1]
// and its keypoint is at (xy[2*i], xy[2*i + 1]) in pixels.
struct Descriptors {
    int rows = 0;
    std::vector<float> data;
    std::vector<float> xy;

    const float *row(int i) const { return &data[(size_t)i * DIM]; }
    float x(int i) const { return xy[(size_t)i * 2]; }
    float y(int i) const { return xy[(size_t)i * 2 + 1]; }
};

// Runs SIFT on an image. If cacheDir is not empty, the result is saved there
// the first time and read back from the file in later calls.
Descriptors getDescriptors(const Image &img, const std::string &cacheDir);

#endif
