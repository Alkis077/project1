#include <chrono>
#include <iostream>
#include <vector>

#include "args.h"
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
        std::cout << "Vocabulary: " << vocab.size() / 128 << " visual words\n";
        std::cout << "First word, first 8 values:";
        for (int j = 0; j < 8; j++)
            std::cout << " " << vocab[j];
        std::cout << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    auto end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << secs << " s\n";
    return 0;
}
