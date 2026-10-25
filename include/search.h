#ifndef SEARCH_H
#define SEARCH_H

#include <string>
#include <vector>

#include "bovw.h"

// Common interface of all search methods (Exact, LSH, Hypercube, IVFFlat, IVFPQ).
class SearchMethod {
public:
    virtual ~SearchMethod() {}

    // name written in the output file
    virtual std::string name() const = 0;

    // builds the index on the base vectors (not counted in the query time)
    virtual void build(const Database &db) = 0;

    // indices (in db.base) of the k closest base images to q, closest first.
    // Approximate methods may return fewer than k.
    virtual std::vector<int> search(const float *q, int k) const = 0;
};

#endif
