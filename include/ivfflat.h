#ifndef IVFFLAT_H
#define IVFFLAT_H

#include <vector>

#include "search.h"

// IVFFlat: k-means on the base vectors splits them into kclusters lists
// (inverted file). A query only looks in the nprobe lists whose centroids are
// closest to it, and compares with every vector in them.
class IVFFlat : public SearchMethod {
public:
    IVFFlat(int kclusters, int nprobe, unsigned seed)
        : kclusters(kclusters), nprobe(nprobe), seed(seed) {}

    std::string name() const override { return "IVFFlat"; }
    void build(const Database &db) override;
    std::vector<int> search(const float *q, int topK) const override;

protected:
    // the nprobe lists (cluster numbers) whose centroids are closest to q
    std::vector<int> closestLists(const float *q) const;

    int kclusters, nprobe;
    unsigned seed;
    const Database *db = nullptr;
    std::vector<float> centroids;          // kclusters * dim
    std::vector<int> assign;               // cluster of every base vector
    std::vector<std::vector<int>> lists;   // base indices in every cluster
};

#endif
