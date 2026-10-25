#ifndef LSH_H
#define LSH_H

#include <cstdint>
#include <utility>
#include <vector>

#include "search.h"

// LSH for Euclidean distance with L hash tables.
// h(p) = floor((p.v + t) / w), v with N(0,1) values, t uniform in [0, w)
// ID(p) = (r_1 h_1(p) + ... + r_k h_k(p)) mod M,  g(p) = ID(p) mod TableSize
// In the bucket g(q) we only compare the points with the same ID as q.
class LSH : public SearchMethod {
public:
    LSH(int k, int L, double w, unsigned seed) : k(k), L(L), w(w), seed(seed) {}

    std::string name() const override { return "LSH"; }
    void build(const Database &db) override;
    std::vector<int> search(const float *q, int topK) const override;

private:
    struct Table {
        std::vector<float> v;     // k random vectors (k * dim)
        std::vector<float> t;     // k shifts
        std::vector<uint64_t> r;  // k random integers
        std::vector<std::vector<std::pair<uint64_t, int>>> buckets;   // (ID, base index)
    };

    uint64_t id(const Table &table, const float *p) const;

    int k, L;
    double w;
    unsigned seed;
    const Database *db = nullptr;
    size_t tableSize = 1;
    std::vector<Table> tables;
};

#endif
