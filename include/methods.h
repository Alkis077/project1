#ifndef METHODS_H
#define METHODS_H

#include <memory>

#include "args.h"
#include "search.h"

// the search method chosen in the arguments (nullptr if it is unknown).
// Used by both search and verify.
std::unique_ptr<SearchMethod> createMethod(const Args &args);

#endif
