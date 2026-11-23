#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "args.h"
#include "bovw.h"
#include "dataset.h"
#include "methods.h"
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

            std::cout << "Query: " << db.queries[q].seq << "\n"
                      << "Candidates: " << cand.images.size() << "\n";
            for (int r = 0; r < TOP; r++)
                std::cout << "Before-Rank-" << r + 1 << ": "
                          << (r < (int)cand.images.size() ? db.base[cand.images[r]].id : "-")
                          << "\n";
        }
        std::cerr << "Queries with fewer than " << args.C << " candidates: " << fewer << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
