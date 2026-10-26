#ifndef IVFPQ_H
#define IVFPQ_H

#include <cstdint>
#include <vector>

#include "ivfflat.h"

// IVFPQ: the same inverted lists as IVFFlat, but every vector is stored as a
// short code. The residual r = x - c(x) is split into M parts and each part is
// replaced by the number of its closest centroid from a codebook of s = 2^nbits
// centroids (k-means on that part of all residuals).
// The query uses a lookup table (ADC): for every list it computes the distance
// of each part of its own residual to every centroid once, and then the
// distance to a stored code is the sum of M table entries.
class IVFPQ : public IVFFlat {
public:
    IVFPQ(int kclusters, int nprobe, int M, int nbits, unsigned seed)
        : IVFFlat(kclusters, nprobe, seed), M(M), nbits(nbits) {}

    std::string name() const override { return "IVFPQ"; }
    void build(const Database &db) override;
    std::vector<int> search(const float *q, int topK) const override;

private:
    int M, nbits;
    int sub = 0;                      // length of every part (dim / M)
    int s = 0;                        // centroids per codebook (2^nbits)
    std::vector<float> codebooks;     // M codebooks, each s * sub floats
    std::vector<uint16_t> codes;      // M numbers for every base vector
};

#endif
