#include "bovw.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "descriptors.h"
#include "kmeans.h"

namespace fs = std::filesystem;

// BoVW vectors of a list of images (images.size() * K floats).
// empty[i] is set when image i has no SIFT descriptors (its vector stays all zeros).
// SIFT extraction runs in parallel, one image per thread.
static void computeVectors(const std::vector<Image> &images, const std::vector<float> &vocab,
                           int K, const std::string &cacheDir, std::vector<float> &vecs,
                           std::vector<bool> &empty)
{
    int n = images.size();
    vecs.assign((size_t)n * K, 0.0f);
    std::vector<char> isEmpty(n, 0);   // vector<bool> is not safe to write from many threads
    std::string error;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        try {
            Descriptors desc = getDescriptors(images[i], cacheDir);
            if (desc.rows == 0) {
                isEmpty[i] = 1;
                continue;
            }
            float *x = &vecs[(size_t)i * K];
            for (int r = 0; r < desc.rows; r++)
                x[nearestCenter(desc.row(r), vocab, K, DIM)] += 1;
            for (int j = 0; j < K; j++)
                x[j] /= desc.rows;
        } catch (const std::exception &e) {
            #pragma omp critical
            error = e.what();
        }
    }
    if (!error.empty())
        throw std::runtime_error(error);

    empty.assign(isEmpty.begin(), isEmpty.end());
}

// appends the first D MIRFlickr images with SIFT descriptors to the base.
// Images are processed in groups of (D - added), so we never compute more
// images than needed, but still process many of them in parallel.
static void addDistractors(Database &db, const std::vector<float> &vocab, int K,
                           const std::string &mirDir, int D, const std::string &cacheDir)
{
    int added = 0;
    int next = 1;
    while (added < D && next <= MIR_TOTAL) {
        std::vector<Image> group;
        while ((int)group.size() < D - added && next <= MIR_TOTAL)
            group.push_back(mirImage(mirDir, next++));

        std::vector<float> vecs;
        std::vector<bool> empty;
        computeVectors(group, vocab, K, cacheDir, vecs, empty);

        for (size_t i = 0; i < group.size(); i++) {
            if (empty[i]) {
                std::cerr << "Skipping " << group[i].id << " (no SIFT descriptors)\n";
                continue;
            }
            db.base.push_back(group[i]);
            db.baseVecs.insert(db.baseVecs.end(), vecs.begin() + i * K,
                               vecs.begin() + (i + 1) * K);
            added++;
        }
        std::cerr << "MIRFlickr: " << added << " / " << D << "\n";
    }
    if (added < D)
        std::cerr << "Warning: only " << added << " MIRFlickr images have SIFT descriptors\n";
}

// cache file format: dim, number of queries, number of base images (ints),
// then for every image its id and sequence (int length + characters) and its vector
static void writeString(std::ofstream &out, const std::string &s)
{
    int len = s.size();
    out.write(reinterpret_cast<const char *>(&len), sizeof(len));
    out.write(s.data(), len);
}

static bool readString(std::ifstream &in, std::string &s)
{
    int len = 0;
    if (!in.read(reinterpret_cast<char *>(&len), sizeof(len)) || len < 0)
        return false;
    s.resize(len);
    return (bool)in.read(&s[0], len);
}

static void writeImages(std::ofstream &out, const std::vector<Image> &imgs,
                        const std::vector<float> &vecs)
{
    for (const Image &img : imgs) {
        writeString(out, img.id);
        writeString(out, img.seq);
    }
    out.write(reinterpret_cast<const char *>(vecs.data()), vecs.size() * sizeof(float));
}

static bool readImages(std::ifstream &in, int count, int dim, std::vector<Image> &imgs,
                       std::vector<float> &vecs)
{
    imgs.resize(count);
    for (Image &img : imgs)
        if (!readString(in, img.id) || !readString(in, img.seq))
            return false;
    vecs.resize((size_t)count * dim);
    return (bool)in.read(reinterpret_cast<char *>(vecs.data()), vecs.size() * sizeof(float));
}

static void saveDatabase(const std::string &file, const Database &db)
{
    fs::create_directories(fs::path(file).parent_path());
    std::string tmp = file + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        int header[3] = {db.dim, (int)db.queries.size(), (int)db.base.size()};
        out.write(reinterpret_cast<const char *>(header), sizeof(header));
        writeImages(out, db.queries, db.queryVecs);
        writeImages(out, db.base, db.baseVecs);
        if (!out)
            throw std::runtime_error("cannot write " + tmp);
    }
    fs::rename(tmp, file);
}

static bool loadDatabase(const std::string &file, int K, Database &db)
{
    std::ifstream in(file, std::ios::binary);
    int header[3];
    if (!in || !in.read(reinterpret_cast<char *>(header), sizeof(header)) || header[0] != K)
        return false;
    db.dim = K;
    return readImages(in, header[1], K, db.queries, db.queryVecs) &&
           readImages(in, header[2], K, db.base, db.baseVecs);
}

Database buildDatabase(const Dataset &ds, const std::vector<float> &vocab, int K,
                       const std::string &mirDir, int D, const std::string &set,
                       const std::string &cacheDir, const std::string &vocabTag)
{
    std::string file;
    if (!cacheDir.empty()) {
        file = cacheDir + "/bovw/" + vocabTag + "_" + set + "_D" + std::to_string(D) + ".bin";
        Database db;
        if (loadDatabase(file, K, db)) {
            std::cerr << "BoVW: read from " << file << "\n";
            return db;
        }
    }

    Database db;
    db.dim = K;
    db.queries = ds.queries;
    db.base = ds.base;

    std::vector<bool> empty;
    computeVectors(db.queries, vocab, K, cacheDir, db.queryVecs, empty);
    computeVectors(db.base, vocab, K, cacheDir, db.baseVecs, empty);
    std::cerr << "BoVW: " << db.queries.size() << " queries, " << db.base.size()
              << " HPatches base images\n";

    addDistractors(db, vocab, K, mirDir, D, cacheDir);

    if (!file.empty()) {
        saveDatabase(file, db);
        std::cerr << "BoVW: saved to " << file << "\n";
    }
    return db;
}
