#include "vocabulary.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "descriptors.h"
#include "kmeans.h"

namespace fs = std::filesystem;

const int KMEANS_MAX_ITER = 100;

// up to S random descriptors of one image. Every image gets its own random
// generator (seed + image index), so the result does not depend on the order
// in which the threads run.
static std::vector<float> sampleDescriptors(const Descriptors &desc, int S, unsigned seed)
{
    std::vector<int> idx(desc.rows);
    std::iota(idx.begin(), idx.end(), 0);

    int take = desc.rows;
    if (desc.rows > S) {
        // partial shuffle: the first S positions become a random choice of S rows
        std::mt19937 rng(seed);
        for (int i = 0; i < S; i++) {
            std::uniform_int_distribution<int> pick(i, desc.rows - 1);
            std::swap(idx[i], idx[pick(rng)]);
        }
        take = S;
    }

    std::vector<float> out;
    out.reserve((size_t)take * DIM);
    for (int i = 0; i < take; i++) {
        const float *r = desc.row(idx[i]);
        out.insert(out.end(), r, r + DIM);
    }
    return out;
}

// The vocabulary depends on the training images, K, S and the seed, so all of
// them go into the file name. The training images (which come from the split
// file) are represented by a hash of their ids.
std::string vocabName(const std::vector<Image> &train, int K, int S, unsigned seed)
{
    uint64_t h = 1469598103934665603ULL;   // FNV-1a hash
    for (const Image &img : train)
        for (char ch : img.id + ";") {
            h ^= (unsigned char)ch;
            h *= 1099511628211ULL;
        }
    char hex[17];
    snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)h);
    return "K" + std::to_string(K) + "_S" + std::to_string(S) +
           "_seed" + std::to_string(seed) + "_train" + hex;
}

static std::string vocabFile(const std::vector<Image> &train, int K, int S,
                             unsigned seed, const std::string &cacheDir)
{
    return cacheDir + "/vocab/" + vocabName(train, K, S, seed) + ".bin";
}

// file format: K (int), then K*128 floats
static bool loadVocabulary(const std::string &file, int K, std::vector<float> &vocab)
{
    std::ifstream in(file, std::ios::binary);
    int k = 0;
    if (!in || !in.read(reinterpret_cast<char *>(&k), sizeof(k)) || k != K)
        return false;
    vocab.resize((size_t)K * DIM);
    return (bool)in.read(reinterpret_cast<char *>(vocab.data()), vocab.size() * sizeof(float));
}

static void saveVocabulary(const std::string &file, int K, const std::vector<float> &vocab)
{
    fs::create_directories(fs::path(file).parent_path());
    std::string tmp = file + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char *>(&K), sizeof(K));
        out.write(reinterpret_cast<const char *>(vocab.data()), vocab.size() * sizeof(float));
        if (!out)
            throw std::runtime_error("cannot write " + tmp);
    }
    fs::rename(tmp, file);
}

static std::vector<float> computeVocabulary(const std::vector<Image> &train, int K, int S,
                                            unsigned seed, const std::string &cacheDir)
{
    int n = train.size();
    std::vector<std::vector<float>> samples(n);
    std::string error;

    // SIFT extraction runs in parallel (one image per thread)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        try {
            Descriptors desc = getDescriptors(train[i], cacheDir);
            samples[i] = sampleDescriptors(desc, S, seed + i);
        } catch (const std::exception &e) {
            #pragma omp critical
            error = e.what();
        }
    }
    if (!error.empty())
        throw std::runtime_error(error);

    std::vector<float> points;
    for (const auto &s : samples)
        points.insert(points.end(), s.begin(), s.end());
    int numPoints = points.size() / DIM;
    std::cerr << "Vocabulary: " << numPoints << " descriptors from " << n
              << " training images\n";

    KMeansResult res = kmeans(points, numPoints, DIM, K, seed, KMEANS_MAX_ITER);
    std::cerr << "Vocabulary: k-means finished after " << res.iterations << " iterations\n";
    return res.centers;
}

std::vector<float> buildVocabulary(const std::vector<Image> &train, int K, int S,
                                   unsigned seed, const std::string &cacheDir)
{
    if (cacheDir.empty())
        return computeVocabulary(train, K, S, seed, cacheDir);

    std::string file = vocabFile(train, K, S, seed, cacheDir);
    std::vector<float> vocab;
    if (loadVocabulary(file, K, vocab)) {
        std::cerr << "Vocabulary: read from " << file << "\n";
        return vocab;
    }
    vocab = computeVocabulary(train, K, S, seed, cacheDir);
    saveVocabulary(file, K, vocab);
    std::cerr << "Vocabulary: saved to " << file << "\n";
    return vocab;
}
