#ifndef BOVW_H
#define BOVW_H

#include <string>
#include <vector>

#include "dataset.h"

// Everything the search methods need: the query and base images and their
// BoVW vectors (dim = K numbers per image, one image after the other).
struct Database {
    int dim = 0;
    std::vector<Image> queries;
    std::vector<Image> base;
    std::vector<float> queryVecs;
    std::vector<float> baseVecs;

    const float *query(int i) const { return &queryVecs[(size_t)i * dim]; }
    const float *baseVec(int i) const { return &baseVecs[(size_t)i * dim]; }
};

// BoVW vector of every image: each SIFT descriptor is assigned to its closest
// visual word, and x_j = h_j / n_I (h_j = descriptors in word j, n_I = all
// descriptors of the image).
// The base gets the HPatches base images of ds and then the first D MIRFlickr
// images that have SIFT descriptors.
// If cacheDir is not empty, the result is saved there and reused when the
// vocabulary (vocabTag), the set and D are the same.
Database buildDatabase(const Dataset &ds, const std::vector<float> &vocab, int K,
                       const std::string &mirDir, int D, const std::string &set,
                       const std::string &cacheDir, const std::string &vocabTag);

#endif
