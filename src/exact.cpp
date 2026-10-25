#include "exact.h"

#include <algorithm>
#include <utility>

#include "kmeans.h"

void ExactSearch::build(const Database &database)
{
    db = &database;   // nothing to build, we only keep the vectors
}

std::vector<int> ExactSearch::search(const float *q, int k) const
{
    int n = db->base.size();

    // (squared distance, index) for every base image.
    // The square root is not needed: it does not change the order.
    std::vector<std::pair<float, int>> dist(n);
    for (int i = 0; i < n; i++)
        dist[i] = {sqDist(q, db->baseVec(i), db->dim), i};

    k = std::min(k, n);
    std::partial_sort(dist.begin(), dist.begin() + k, dist.end());

    std::vector<int> result(k);
    for (int i = 0; i < k; i++)
        result[i] = dist[i].second;
    return result;
}
