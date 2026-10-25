#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include <cstdint>
#include <vector>

#include "search.h"

// Randomized projection to the hypercube {0,1}^kproj.
// For every i: h_i(p) = floor((p.v_i + t_i) / w) as in LSH, and f_i maps each
// value of h_i to 0 or 1 at random. The bits f_1 ... f_kproj give the vertex of p.
// The query checks its own vertex and then the vertices with Hamming distance
// 1, 2, ... until it has checked M points or visited `probes` vertices.
class Hypercube : public SearchMethod {
public:
    Hypercube(int kproj, double w, int M, int probes, unsigned seed)
        : kproj(kproj), w(w), M(M), probes(probes), seed(seed) {}

    std::string name() const override { return "Hypercube"; }
    void build(const Database &db) override;
    std::vector<int> search(const float *q, int topK) const override;

private:
    int bit(int i, long long h) const;
    uint32_t vertex(const float *p) const;

    int kproj;
    double w;
    int M, probes;
    unsigned seed;
    const Database *db = nullptr;
    std::vector<float> v;                    // kproj random vectors (kproj * dim)
    std::vector<float> t;                    // kproj shifts
    std::vector<std::vector<int>> vertices;  // base indices in every vertex
    std::vector<uint32_t> probeOrder;        // all bit masks, fewer 1s first
};

#endif
