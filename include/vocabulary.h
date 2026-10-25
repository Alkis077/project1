#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <string>
#include <vector>

#include "dataset.h"

// Builds the visual vocabulary: up to S random SIFT descriptors from every
// training image, then k-means with K centers on all of them.
// Returns the K visual words (K * 128 floats).
std::vector<float> buildVocabulary(const std::vector<Image> &train, int K, int S,
                                   unsigned seed, const std::string &cacheDir);

// short name that describes a vocabulary (K, S, seed and the training images),
// used for the names of cache files that depend on it
std::string vocabName(const std::vector<Image> &train, int K, int S, unsigned seed);

#endif
