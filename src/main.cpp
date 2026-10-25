#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "args.h"
#include "bovw.h"
#include "dataset.h"
#include "evaluate.h"
#include "exact.h"
#include "hypercube.h"
#include "lsh.h"
#include "vocabulary.h"

static std::unique_ptr<SearchMethod> createMethod(const Args &args)
{
    if (args.method == "exact")
        return std::make_unique<ExactSearch>();
    if (args.method == "lsh")
        return std::make_unique<LSH>(args.k, args.L, args.w, args.seed);
    if (args.method == "hypercube")
        return std::make_unique<Hypercube>(args.kproj, args.w, args.hcM, args.probes, args.seed);
    return nullptr;
}

int main(int argc, char **argv)
{
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage(argv[0]);
        return 1;
    }

    auto start = std::chrono::steady_clock::now();

    try {
        std::unique_ptr<SearchMethod> method = createMethod(args);
        if (!method) {
            std::cerr << "Method " << args.method << " is not implemented yet\n";
            return 1;
        }

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
        evaluate(*method, db, args.outFile);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    auto end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();
    std::cout << "Total time: " << secs << " s\n";
    return 0;
}
