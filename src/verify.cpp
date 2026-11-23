#include <iostream>

#include "args.h"

int main(int argc, char **argv)
{
    Args args;
    if (!parseArgs(argc, argv, args, true)) {
        printUsage(argv[0], true);
        return 1;
    }

    std::cerr << "Method: " << args.method << ", C: " << args.C << ", rho: " << args.rho
              << ", tau: " << args.tau << ", T: " << args.Tmax << ", p: " << args.p
              << ", adaptive: " << args.adaptive << "\n";
    return 0;
}
