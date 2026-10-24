#include "args.h"

#include <iostream>
#include <string>

void printUsage(const char *prog)
{
    std::cerr << "Usage: " << prog << " -hp <path> -mir <path> -split <file>"
              << " [-vocab K] [-S S] [-D D] [-set validation|test] [-o file] [-seed n] [-cache dir]"
              << " -exact|-lsh|-hypercube|-ivfflat|-ivfpq [method parameters]\n";
}

bool parseArgs(int argc, char **argv, Args &args)
{
    int M = -1;   // -M means different things for hypercube and ivfpq

    for (int i = 1; i < argc; i++) {
        std::string opt = argv[i];

        if (opt == "-exact" || opt == "-lsh" || opt == "-hypercube" ||
            opt == "-ivfflat" || opt == "-ivfpq") {
            if (!args.method.empty()) {
                std::cerr << "Only one search method can be given\n";
                return false;
            }
            args.method = opt.substr(1);
            continue;
        }

        // every other option needs a value after it
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << opt << "\n";
            return false;
        }
        std::string val = argv[++i];

        try {
            if (opt == "-hp") args.hpDir = val;
            else if (opt == "-mir") args.mirDir = val;
            else if (opt == "-split") args.splitFile = val;
            else if (opt == "-set") args.set = val;
            else if (opt == "-o") args.outFile = val;
            else if (opt == "-cache") args.cacheDir = val;
            else if (opt == "-vocab") args.vocab = std::stoi(val);
            else if (opt == "-S") args.S = std::stoi(val);
            else if (opt == "-D") args.D = std::stoi(val);
            else if (opt == "-seed") args.seed = std::stoul(val);
            else if (opt == "-k") args.k = std::stoi(val);
            else if (opt == "-L") args.L = std::stoi(val);
            else if (opt == "-w") args.w = std::stod(val);
            else if (opt == "-kproj") args.kproj = std::stoi(val);
            else if (opt == "-M") M = std::stoi(val);
            else if (opt == "-probes") args.probes = std::stoi(val);
            else if (opt == "-kclusters") args.kclusters = std::stoi(val);
            else if (opt == "-nprobe") args.nprobe = std::stoi(val);
            else if (opt == "-nbits") args.nbits = std::stoi(val);
            else {
                std::cerr << "Unknown option " << opt << "\n";
                return false;
            }
        } catch (const std::exception &) {
            std::cerr << "Bad value for " << opt << ": " << val << "\n";
            return false;
        }
    }

    if (M != -1) {
        if (args.method == "ivfpq") args.pqM = M;
        else args.hcM = M;
    }

    if (args.hpDir.empty() || args.mirDir.empty() || args.splitFile.empty()) {
        std::cerr << "-hp, -mir and -split are required\n";
        return false;
    }
    if (args.method.empty()) {
        std::cerr << "No search method given\n";
        return false;
    }
    if (args.set != "validation" && args.set != "test") {
        std::cerr << "-set must be validation or test\n";
        return false;
    }
    if (args.method == "ivfpq" && args.vocab % args.pqM != 0) {
        std::cerr << "For ivfpq, vocab must be divisible by M\n";
        return false;
    }
    return true;
}
