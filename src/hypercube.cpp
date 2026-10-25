#include "hypercube.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include "kmeans.h"

// f_i: every value of h_i gets 0 or 1 at random, but always the same bit for
// the same value. Instead of storing a table of random bits we mix
// (seed, i, h) into a pseudo-random number and keep its lowest bit.
int Hypercube::bit(int i, long long h) const
{
    uint64_t x = (uint64_t)h * 0x9E3779B97F4A7C15ULL + (uint64_t)(i + 1) * 0xBF58476D1CE4E5B9ULL + seed;
    x ^= x >> 31;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 29;
    return x & 1;
}

uint32_t Hypercube::vertex(const float *p) const
{
    int dim = db->dim;
    uint32_t vert = 0;
    for (int i = 0; i < kproj; i++) {
        const float *vi = &v[(size_t)i * dim];
        double dot = 0;
        for (int j = 0; j < dim; j++)
            dot += p[j] * vi[j];
        long long h = (long long)std::floor((dot + t[i]) / w);
        vert = (vert << 1) | bit(i, h);
    }
    return vert;
}

void Hypercube::build(const Database &database)
{
    if (kproj < 1 || kproj > 24)
        throw std::runtime_error("hypercube: kproj must be between 1 and 24");

    db = &database;
    int dim = db->dim;

    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> shift(0.0f, (float)w);
    v.resize((size_t)kproj * dim);
    for (float &x : v)
        x = normal(rng);
    t.resize(kproj);
    for (float &x : t)
        x = shift(rng);

    uint32_t numVertices = 1u << kproj;
    vertices.assign(numVertices, {});
    // The base has the HPatches images first and MIRFlickr after them. Since the
    // search stops after M points, inserting them in that order would favour the
    // HPatches images (the correct answers). So they are inserted in random order.
    std::vector<int> order(db->base.size());
    for (int i = 0; i < (int)order.size(); i++)
        order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    for (int i : order)
        vertices[vertex(db->baseVec(i))].push_back(i);

    size_t used = 0, largest = 0;
    for (const auto &vert : vertices) {
        if (!vert.empty())
            used++;
        largest = std::max(largest, vert.size());
    }
    std::cerr << "Hypercube: " << used << " of " << numVertices
              << " vertices have points, largest has " << largest << "\n";

    // q XOR mask is a vertex at Hamming distance popcount(mask) from q,
    // so sorting the masks by number of 1s gives the probing order
    probeOrder.resize(numVertices);
    for (uint32_t m = 0; m < numVertices; m++)
        probeOrder[m] = m;
    std::stable_sort(probeOrder.begin(), probeOrder.end(), [](uint32_t a, uint32_t b) {
        return std::bitset<32>(a).count() < std::bitset<32>(b).count();
    });
}

std::vector<int> Hypercube::search(const float *q, int topK) const
{
    uint32_t start = vertex(q);
    std::vector<std::pair<float, int>> candidates;

    int visited = 0;
    for (uint32_t mask : probeOrder) {
        if (visited == probes || (int)candidates.size() >= M)
            break;
        visited++;
        for (int idx : vertices[start ^ mask]) {
            if ((int)candidates.size() >= M)
                break;
            candidates.push_back({sqDist(q, db->baseVec(idx), db->dim), idx});
        }
    }

    int k = std::min<int>(topK, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());

    std::vector<int> result(k);
    for (int i = 0; i < k; i++)
        result[i] = candidates[i].second;
    return result;
}
