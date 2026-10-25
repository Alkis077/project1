#include "lsh.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "kmeans.h"

static const uint64_t M = 4294967291ULL;   // 2^32 - 5, a prime
static const int TABLE_DIVISOR = 8;        // TableSize = n / 8

uint64_t LSH::id(const Table &table, const float *p) const
{
    int dim = db->dim;
    uint64_t sum = 0;
    for (int i = 0; i < k; i++) {
        const float *v = &table.v[(size_t)i * dim];
        double dot = 0;
        for (int j = 0; j < dim; j++)
            dot += p[j] * v[j];
        long long h = (long long)std::floor((dot + table.t[i]) / w);

        // (r_i * h_i) mod M, with h_i possibly negative
        long long hm = h % (long long)M;
        if (hm < 0)
            hm += M;
        sum = (sum + table.r[i] * (uint64_t)hm % M) % M;
    }
    return sum;
}

void LSH::build(const Database &database)
{
    db = &database;
    int n = db->base.size();
    int dim = db->dim;
    tableSize = std::max(1, n / TABLE_DIVISOR);

    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> shift(0.0f, (float)w);
    std::uniform_int_distribution<uint64_t> randomR(1, M - 1);

    tables.assign(L, Table());
    for (Table &table : tables) {
        table.v.resize((size_t)k * dim);
        for (float &x : table.v)
            x = normal(rng);
        table.t.resize(k);
        for (float &x : table.t)
            x = shift(rng);
        table.r.resize(k);
        for (uint64_t &x : table.r)
            x = randomR(rng);

        table.buckets.assign(tableSize, {});
        for (int i = 0; i < n; i++) {
            uint64_t ID = id(table, db->baseVec(i));
            table.buckets[ID % tableSize].push_back({ID, i});
        }
    }
}

std::vector<int> LSH::search(const float *q, int topK) const
{
    std::vector<char> seen(db->base.size(), 0);   // a point can be in the same bucket in many tables
    std::vector<std::pair<float, int>> candidates;

    for (const Table &table : tables) {
        uint64_t ID = id(table, q);
        for (const auto &entry : table.buckets[ID % tableSize]) {
            if (entry.first != ID || seen[entry.second])
                continue;
            seen[entry.second] = 1;
            candidates.push_back({sqDist(q, db->baseVec(entry.second), db->dim), entry.second});
        }
    }

    int k = std::min<int>(topK, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());

    std::vector<int> result(k);
    for (int i = 0; i < k; i++)
        result[i] = candidates[i].second;
    return result;
}
