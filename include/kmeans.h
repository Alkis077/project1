#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

// squared Euclidean distance between two vectors of length dim
inline float sqDist(const float *a, const float *b, int dim)
{
    float s = 0;
    for (int j = 0; j < dim; j++) {
        float d = a[j] - b[j];
        s += d * d;
    }
    return s;
}

// index of the center (out of K, each of length dim) that is closest to p
int nearestCenter(const float *p, const std::vector<float> &centers, int K, int dim);

struct KMeansResult {
    std::vector<float> centers;   // K vectors of length dim, one after the other
    std::vector<int> assign;      // cluster of every point
    int iterations = 0;
};

// k-means++ initialization and then Lloyd's algorithm.
// points holds n vectors of length dim, one after the other.
// Stops when no point changes cluster, or after maxIter iterations.
KMeansResult kmeans(const std::vector<float> &points, int n, int dim, int K,
                    unsigned seed, int maxIter);

#endif
