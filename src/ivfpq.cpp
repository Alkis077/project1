#include "ivfpq.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "kmeans.h"

const int PQ_KMEANS_MAX_ITER = 100;

void IVFPQ::build(const Database &database)
{
    IVFFlat::build(database);   // coarse k-means and the inverted lists

    int n = db->base.size();
    int dim = db->dim;
    if (dim % M != 0)
        throw std::runtime_error("ivfpq: vector length must be divisible by M");
    if (nbits < 1 || nbits > 16)
        throw std::runtime_error("ivfpq: nbits must be between 1 and 16");
    sub = dim / M;
    s = 1 << nbits;

    // residuals r = x - c(x)
    std::vector<float> residuals((size_t)n * dim);
    for (int i = 0; i < n; i++) {
        const float *x = db->baseVec(i);
        const float *c = &centroids[(size_t)assign[i] * dim];
        for (int j = 0; j < dim; j++)
            residuals[(size_t)i * dim + j] = x[j] - c[j];
    }

    codebooks.resize((size_t)M * s * sub);
    codes.resize((size_t)n * M);
    std::vector<float> part((size_t)n * sub);
    for (int m = 0; m < M; m++) {
        // part m of every residual
        for (int i = 0; i < n; i++)
            for (int j = 0; j < sub; j++)
                part[(size_t)i * sub + j] = residuals[(size_t)i * dim + m * sub + j];

        KMeansResult res = kmeans(part, n, sub, s, seed + m + 1, PQ_KMEANS_MAX_ITER);
        std::copy(res.centers.begin(), res.centers.end(),
                  codebooks.begin() + (size_t)m * s * sub);
        for (int i = 0; i < n; i++)
            codes[(size_t)i * M + m] = res.assign[i];
    }
    std::cerr << "IVFPQ: " << M << " parts of length " << sub << ", " << s
              << " centroids each\n";
}

std::vector<int> IVFPQ::search(const float *q, int topK) const
{
    int dim = db->dim;
    std::vector<float> rq(dim);
    std::vector<float> table((size_t)M * s);
    std::vector<std::pair<float, int>> candidates;

    for (int c : closestLists(q)) {
        // residual of the query for this list
        const float *centroid = &centroids[(size_t)c * dim];
        for (int j = 0; j < dim; j++)
            rq[j] = q[j] - centroid[j];

        // table[m][k] = distance^2 of part m of rq to centroid k of codebook m
        for (int m = 0; m < M; m++)
            for (int k = 0; k < s; k++)
                table[(size_t)m * s + k] =
                    sqDist(&rq[m * sub], &codebooks[((size_t)m * s + k) * sub], sub);

        // approximate distance of every vector in the list: sum of M table entries
        for (int idx : lists[c]) {
            const uint16_t *code = &codes[(size_t)idx * M];
            float d = 0;
            for (int m = 0; m < M; m++)
                d += table[(size_t)m * s + code[m]];
            candidates.push_back({d, idx});
        }
    }

    int k = std::min<int>(topK, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());

    std::vector<int> result(k);
    for (int i = 0; i < k; i++)
        result[i] = candidates[i].second;
    return result;
}
