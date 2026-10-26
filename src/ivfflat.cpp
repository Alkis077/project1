#include "ivfflat.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "kmeans.h"

const int IVF_KMEANS_MAX_ITER = 100;

void IVFFlat::build(const Database &database)
{
    db = &database;
    int n = db->base.size();
    if (kclusters <= 0)
        kclusters = (int)std::ceil(std::sqrt((double)n));
    nprobe = std::min(nprobe, kclusters);

    KMeansResult res = kmeans(db->baseVecs, n, db->dim, kclusters, seed, IVF_KMEANS_MAX_ITER);
    centroids = res.centers;
    assign = res.assign;

    lists.assign(kclusters, {});
    for (int i = 0; i < n; i++)
        lists[assign[i]].push_back(i);

    std::cerr << "IVF: " << kclusters << " clusters, k-means " << res.iterations
              << " iterations\n";
}

std::vector<int> IVFFlat::closestLists(const float *q) const
{
    std::vector<std::pair<float, int>> dist(kclusters);
    for (int c = 0; c < kclusters; c++)
        dist[c] = {sqDist(q, &centroids[(size_t)c * db->dim], db->dim), c};
    std::partial_sort(dist.begin(), dist.begin() + nprobe, dist.end());

    std::vector<int> result(nprobe);
    for (int i = 0; i < nprobe; i++)
        result[i] = dist[i].second;
    return result;
}

std::vector<int> IVFFlat::search(const float *q, int topK) const
{
    std::vector<std::pair<float, int>> candidates;
    for (int c : closestLists(q))
        for (int idx : lists[c])
            candidates.push_back({sqDist(q, db->baseVec(idx), db->dim), idx});

    int k = std::min<int>(topK, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());

    std::vector<int> result(k);
    for (int i = 0; i < k; i++)
        result[i] = candidates[i].second;
    return result;
}
