#ifndef ARGS_H
#define ARGS_H

#include <string>

struct Args {
    // common
    std::string hpDir;
    std::string mirDir;
    std::string splitFile;
    std::string set = "validation";
    std::string outFile = "output.txt";
    std::string cacheDir;        // -cache <dir>: save/reuse SIFT, vocabulary and BoVW there. Empty = no cache
    std::string method;          // exact, lsh, hypercube, ivfflat, ivfpq
    int vocab = 256;
    int S = 500;
    int D = 1000;
    unsigned seed = 1;

    // lsh
    int k = 4;
    int L = 5;
    double w = 0.05;             // also used by hypercube

    // hypercube
    int kproj = 10;
    int hcM = 500;
    int probes = 10;

    // ivfflat / ivfpq
    int kclusters = 0;           // 0 means ceil(sqrt(n))
    int nprobe = 5;
    int pqM = 16;
    int nbits = 8;

    // verify (2nd assignment)
    int C = 50;                  // candidates from the 1st stage
    double rho = 0.8;            // Lowe ratio
    double tau = 3;              // inlier threshold (pixels)
    int Tmax = 2000;             // -T: maximum RANSAC trials
    double p = 0.99;             // probability for the adaptive limit
    bool adaptive = true;        // -adaptive 0|1
};

// returns false (and prints a message) if the arguments are wrong.
// verify = true: also accepts the options of verify, and -kclusters is
// required for ivfflat and ivfpq
bool parseArgs(int argc, char **argv, Args &args, bool verify = false);

void printUsage(const char *prog, bool verify = false);

#endif
