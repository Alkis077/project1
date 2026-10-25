#ifndef EVALUATE_H
#define EVALUATE_H

#include <string>

#include "bovw.h"
#include "search.h"

const int TOP_K = 10;       // results returned for every query
const int RELEVANT = 5;     // images 2..6 of the query's sequence

// Runs every query with the given (already built) method, measures the search
// time, computes Recall@5, Recall@10 and AP@10, writes the output file and
// prints a summary (overall, illumination, viewpoint).
void evaluate(const SearchMethod &method, const Database &db, const std::string &outFile);

#endif
