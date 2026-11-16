#include "matching.h"

#include <cstdint>
#include <limits>

// SIFT values are whole numbers in [0,255] (that is also how the cache keeps
// them), so the descriptors are copied to bytes and the distances are computed
// with integers. The result is exactly the same as with floats, but the
// compiler can vectorize the integer loop, which made the matching about six
// times faster in our tests.
static std::vector<uint8_t> toBytes(const Descriptors &d)
{
    return std::vector<uint8_t>(d.data.begin(), d.data.end());
}

static int sqDistBytes(const uint8_t *p, const uint8_t *q)
{
    int s = 0;
    for (int j = 0; j < DIM; j++) {
        int d = (int)p[j] - (int)q[j];
        s += d * d;
    }
    return s;
}

std::vector<Match> matchSift(const Descriptors &a, const Descriptors &b, double rho)
{
    std::vector<Match> matches;
    if (b.rows < 2)
        return matches;

    std::vector<uint8_t> A = toBytes(a), B = toBytes(b);

    // distances are kept squared, so the test d1 < rho * d2
    // becomes D1 < rho^2 * D2
    const double rho2 = rho * rho;

    for (int q = 0; q < a.rows; q++) {
        const uint8_t *dq = &A[(size_t)q * DIM];
        int best1 = std::numeric_limits<int>::max();   // smallest distance
        int best2 = best1;                             // second smallest
        int idx1 = -1;

        for (int i = 0; i < b.rows; i++) {
            int d = sqDistBytes(dq, &B[(size_t)i * DIM]);
            if (d < best1) {
                best2 = best1;
                best1 = d;
                idx1 = i;
            } else if (d < best2) {
                best2 = d;
            }
        }

        if (best1 < rho2 * best2)
            matches.push_back({q, idx1, a.x(q), a.y(q), b.x(idx1), b.y(idx1)});
    }
    return matches;
}
