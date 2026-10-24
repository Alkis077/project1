#include "kmeans.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

// k-means++: first center is a random point, every next center is a point
// chosen with probability proportional to D(x)^2, where D(x) is the distance
// of x from its closest center so far.
static std::vector<float> initCenters(const std::vector<float> &points, int n, int dim,
                                      int K, std::mt19937 &rng)
{
    std::vector<float> centers((size_t)K * dim);
    auto setCenter = [&](int c, int p) {
        for (int j = 0; j < dim; j++)
            centers[(size_t)c * dim + j] = points[(size_t)p * dim + j];
    };

    std::uniform_int_distribution<int> anyPoint(0, n - 1);
    setCenter(0, anyPoint(rng));

    // D(x)^2 for every point
    std::vector<double> d2(n);
    for (int i = 0; i < n; i++)
        d2[i] = sqDist(&points[(size_t)i * dim], &centers[0], dim);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int c = 1; c < K; c++) {
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += d2[i];

        int chosen;
        if (sum == 0) {
            chosen = anyPoint(rng);   // all points are already on a center
        } else {
            // walk the partial sums until we pass a random value in [0, sum)
            double r = uniform(rng) * sum;
            chosen = n - 1;
            double acc = 0;
            for (int i = 0; i < n; i++) {
                acc += d2[i];
                if (acc > r) {
                    chosen = i;
                    break;
                }
            }
        }
        setCenter(c, chosen);

        const float *newCenter = &centers[(size_t)c * dim];
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double d = sqDist(&points[(size_t)i * dim], newCenter, dim);
            if (d < d2[i])
                d2[i] = d;
        }
    }
    return centers;
}

static int nearest(const float *p, const std::vector<float> &centers, int K, int dim)
{
    int best = 0;
    float bestDist = sqDist(p, &centers[0], dim);
    for (int c = 1; c < K; c++) {
        float d = sqDist(p, &centers[(size_t)c * dim], dim);
        if (d < bestDist) {
            bestDist = d;
            best = c;
        }
    }
    return best;
}

KMeansResult kmeans(const std::vector<float> &points, int n, int dim, int K,
                    unsigned seed, int maxIter)
{
    if (n < K)
        throw std::runtime_error("k-means: fewer points than clusters");

    std::mt19937 rng(seed);
    KMeansResult res;
    res.centers = initCenters(points, n, dim, K, rng);
    res.assign.assign(n, -1);

    std::vector<double> sums((size_t)K * dim);
    std::vector<int> counts(K);

    for (int it = 1; it <= maxIter; it++) {
        // assignment step: every point goes to its closest center.
        // Points are independent, so they are split between the threads.
        int changes = 0;
        #pragma omp parallel for reduction(+ : changes)
        for (int i = 0; i < n; i++) {
            int c = nearest(&points[(size_t)i * dim], res.centers, K, dim);
            if (c != res.assign[i]) {
                res.assign[i] = c;
                changes++;
            }
        }
        res.iterations = it;
        std::cerr << "k-means iteration " << it << ": " << changes << " points changed\n";
        if (changes == 0)
            break;

        // update step: every center becomes the mean of its points
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (int i = 0; i < n; i++) {
            int c = res.assign[i];
            counts[c]++;
            for (int j = 0; j < dim; j++)
                sums[(size_t)c * dim + j] += points[(size_t)i * dim + j];
        }
        for (int c = 0; c < K; c++) {
            if (counts[c] == 0)
                continue;   // empty cluster: keep the old center
            for (int j = 0; j < dim; j++)
                res.centers[(size_t)c * dim + j] = sums[(size_t)c * dim + j] / counts[c];
        }
    }
    return res;
}
