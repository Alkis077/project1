#ifndef EXACT_H
#define EXACT_H

#include "search.h"

// Exact search: compares the query with every base vector.
class ExactSearch : public SearchMethod {
public:
    std::string name() const override { return "Exact"; }
    void build(const Database &db) override;
    std::vector<int> search(const float *q, int k) const override;

private:
    const Database *db = nullptr;
};

#endif
