#include "methods.h"

#include "exact.h"
#include "hypercube.h"
#include "ivfflat.h"
#include "ivfpq.h"
#include "lsh.h"

std::unique_ptr<SearchMethod> createMethod(const Args &args)
{
    if (args.method == "exact")
        return std::make_unique<ExactSearch>();
    if (args.method == "lsh")
        return std::make_unique<LSH>(args.k, args.L, args.w, args.seed);
    if (args.method == "hypercube")
        return std::make_unique<Hypercube>(args.kproj, args.w, args.hcM, args.probes, args.seed);
    if (args.method == "ivfflat")
        return std::make_unique<IVFFlat>(args.kclusters, args.nprobe, args.seed);
    if (args.method == "ivfpq")
        return std::make_unique<IVFPQ>(args.kclusters, args.nprobe, args.pqM, args.nbits,
                                       args.seed);
    return nullptr;
}
