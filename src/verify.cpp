#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "bovw.h"
#include "dataset.h"
#include "descriptors.h"
#include "matching.h"
#include "methods.h"
#include "ransac.h"
#include "vocabulary.h"

const int TOP = 10;   // images returned for every query

// 1st stage for one query: the C closest base images with the method of the
// 1st assignment (indices in db.base, closest first) and the time it took.
struct Candidates {
    std::vector<int> images;
    double timeMs = 0;
};

static Candidates findCandidates(const SearchMethod &method, const Database &db, int q, int C)
{
    Candidates c;
    auto start = std::chrono::steady_clock::now();
    c.images = method.search(db.query(q), C);
    auto end = std::chrono::steady_clock::now();
    c.timeMs = std::chrono::duration<double, std::milli>(end - start).count();
    return c;
}

// SIFT keypoints and descriptors of the candidates (read from the cache, or
// computed the first time). This is not timed, like the SIFT extraction.
static std::vector<Descriptors> loadCandidates(const Database &db, const std::vector<int> &images,
                                               const std::string &cacheDir)
{
    std::vector<Descriptors> desc(images.size());
    std::string error;
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < (int)images.size(); c++) {
        // an exception must not leave the parallel loop, so it is kept and
        // thrown again after the loop
        try {
            desc[c] = getDescriptors(db.base[images[c]], cacheDir);
        } catch (const std::exception &e) {
            #pragma omp critical
            error = e.what();
        }
    }
    if (!error.empty())
        throw std::runtime_error(error);
    return desc;
}

// 2nd stage result for one candidate
struct Verification {
    int image = 0;            // index in db.base
    int matches = 0;          // M: matches accepted by the Lowe test
    bool ok = false;          // RANSAC found a valid homography
    int inliers = 0;          // final inliers (0 if not ok)
    double medianError = 0;   // median error of the final inliers
    Eigen::Matrix3d H;
};

// SIFT matching and RANSAC between query q and each of its candidates.
// Every pair (q, candidate) gets its own random generator, made from the seed
// and the two images, so the result does not depend on the order in which the
// threads run the candidates.
static std::vector<Verification> verifyCandidates(const Descriptors &query,
                                                  const std::vector<Descriptors> &desc,
                                                  const std::vector<int> &images, int q,
                                                  const Args &args)
{
    RansacParams params;
    params.tau = args.tau;
    params.Tmax = args.Tmax;
    params.p = args.p;
    params.adaptive = args.adaptive;

    std::vector<Verification> ver(images.size());
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < (int)images.size(); c++) {
        Verification &v = ver[c];
        v.image = images[c];

        std::vector<Match> matches = matchSift(query, desc[c], args.rho);
        v.matches = matches.size();

        std::seed_seq seq{args.seed, (unsigned)q, (unsigned)images[c]};
        std::mt19937 rng(seq);
        RansacResult r = ransac(matches, params, rng);
        if (r.ok) {
            v.ok = true;
            v.inliers = r.inliers.size();
            v.medianError = r.medianError;
            v.H = r.H;
        }
    }
    return ver;
}

int main(int argc, char **argv)
{
    Args args;
    if (!parseArgs(argc, argv, args, true)) {
        printUsage(argv[0], true);
        return 1;
    }

    try {
        std::unique_ptr<SearchMethod> method = createMethod(args);

        // same data, vocabulary, BoVW vectors and index as in the 1st assignment
        Dataset ds = loadDataset(args.hpDir, args.splitFile, args.set);
        if (ds.queries.empty()) {
            std::cerr << "No queries found for set " << args.set << "\n";
            return 1;
        }
        std::vector<float> vocab = buildVocabulary(ds.train, args.vocab, args.S,
                                                   args.seed, args.cacheDir);
        Database db = buildDatabase(ds, vocab, args.vocab, args.mirDir, args.D, args.set,
                                    args.cacheDir,
                                    vocabName(ds.train, args.vocab, args.S, args.seed));
        std::cerr << "Queries: " << db.queries.size() << ", base: " << db.base.size() << "\n";
        method->build(db);

        int fewer = 0;   // queries with fewer than C candidates
        for (size_t q = 0; q < db.queries.size(); q++) {
            Candidates cand = findCandidates(*method, db, q, args.C);
            if ((int)cand.images.size() < args.C)
                fewer++;

            // reading the descriptors is not timed
            Descriptors query = getDescriptors(db.queries[q], args.cacheDir);
            std::vector<Descriptors> desc = loadCandidates(db, cand.images, args.cacheDir);

            auto start = std::chrono::steady_clock::now();
            std::vector<Verification> ver = verifyCandidates(query, desc, cand.images, q, args);
            auto end = std::chrono::steady_clock::now();
            double verMs = std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << "Query: " << db.queries[q].seq << "\n"
                      << "Candidates: " << cand.images.size() << "\n"
                      << "Times: candidates " << cand.timeMs << " ms, verification "
                      << verMs << " ms\n";
            for (int r = 0; r < TOP && r < (int)ver.size(); r++) {
                const Verification &v = ver[r];
                std::cout << "Before-Rank-" << r + 1 << ": " << db.base[v.image].id
                          << "  matches " << v.matches << "  inliers " << v.inliers
                          << "  median ";
                if (v.ok)
                    std::cout << v.medianError << "\n";
                else
                    std::cout << "NA\n";
            }
        }
        std::cerr << "Queries with fewer than " << args.C << " candidates: " << fewer << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
