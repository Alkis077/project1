#include <chrono>
#include <iostream>
#include <vector>

#include "args.h"
#include "bovw.h"
#include "dataset.h"
#include "vocabulary.h"

int main(int argc, char **argv)
{
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage(argv[0]);
        return 1;
    }

    auto start = std::chrono::steady_clock::now();

    try {
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

        std::cout << "Queries: " << db.queries.size() << ", base: " << db.base.size()
                  << ", vector length: " << db.dim << "\n";

        // every vector should sum to 1 (h_j / n_I over all words)
        const float *x = db.query(0);
        float sum = 0;
        int nonZero = 0;
        for (int j = 0; j < db.dim; j++) {
            sum += x[j];
            if (x[j] > 0) nonZero++;
        }
        std::cout << "Query " << db.queries[0].id << ": sum " << sum << ", "
                  << nonZero << " of " << db.dim << " words appear\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    auto end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << secs << " s\n";
    return 0;
}
