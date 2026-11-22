// Checks of the DLT and RANSAC code with point correspondences whose
// homography is known (no images, no SIFT matching).

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "homography.h"
#include "ransac.h"

static int failed = 0;

static void check(bool ok, const char *what)
{
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        failed++;
}

// matches (x, y) -> G (x, y)
static std::vector<Match> makeMatches(const Eigen::Matrix3d &G,
                                      const std::vector<std::pair<double, double>> &pts)
{
    std::vector<Match> m;
    for (const auto &p : pts) {
        double u, v;
        project(G, p.first, p.second, u, v);
        m.push_back({0, 0, p.first, p.second, u, v});
    }
    return m;
}

// H and G are the same homography if they agree up to scale
static bool sameHomography(const Eigen::Matrix3d &H, const Eigen::Matrix3d &G)
{
    Eigen::Matrix3d a = H / H.norm(), b = G / G.norm();
    if (a(2, 2) * b(2, 2) < 0)
        b = -b;
    return (a - b).norm() < 1e-9;
}

static double maxError(const Eigen::Matrix3d &H, const std::vector<Match> &m)
{
    double e = 0;
    for (const Match &x : m)
        e = std::max(e, reprojectionError(H, x));
    return e;
}

static std::vector<int> all(int n)
{
    std::vector<int> idx;
    for (int i = 0; i < n; i++)
        idx.push_back(i);
    return idx;
}

int main()
{
    const std::vector<std::pair<double, double>> square = {{0, 0}, {100, 0}, {100, 80}, {0, 80}};
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity(), shift, persp;
    shift << 1, 0, 20, 0, 1, 10, 0, 0, 1;
    persp << 1, 0, 20, 0, 1, 10, 0.001, 0, 1;
    Eigen::Matrix3d H;

    // DLT
    std::vector<Match> m = makeMatches(I, square);
    check(estimateHomography(m, all(4), H) && sameHomography(H, I) && maxError(H, m) < 1e-9,
          "DLT: identity from 4 points");

    m = makeMatches(shift, square);
    check(estimateHomography(m, all(4), H) && sameHomography(H, shift) && maxError(H, m) < 1e-9,
          "DLT: shift by (20, 10) from 4 points");

    m = makeMatches(persp, {{0, 0}, {100, 0}, {100, 80}, {0, 80}, {37, 12}, {250, 300},
                            {500, 20}, {640, 480}, {5, 400}, {321, 123}});
    check(estimateHomography(m, all(m.size()), H) && sameHomography(H, persp) &&
          maxError(H, m) < 1e-9, "DLT: homography of the course example from 10 points");

    double u, v;
    check(project(persp, 100, 50, u, v) && std::fabs(u - 109.0909) < 1e-3 &&
          std::fabs(v - 54.5455) < 1e-3, "projection: (100, 50) -> (109.09, 54.55)");

    // degenerate samples and invalid homographies
    int idx[4] = {0, 1, 2, 3};
    m = makeMatches(shift, {{0, 0}, {100, 0}, {200, 0}, {0, 80}});
    check(degenerateSample(m, idx), "degenerate: three collinear points");
    m = makeMatches(shift, {{0, 0}, {100, 0}, {0, 0}, {0, 80}});
    check(degenerateSample(m, idx), "degenerate: repeated point");
    m = {{0, 0, 0, 0, 5, 5}, {0, 0, 100, 0, 5, 5}, {0, 0, 100, 80, 50, 60}, {0, 0, 0, 80, 9, 70}};
    check(degenerateSample(m, idx), "degenerate: repeated point in the second image only");
    m = makeMatches(shift, square);
    check(!degenerateSample(m, idx), "not degenerate: corners of a rectangle");

    Eigen::Matrix3d singular;
    singular << 1, 2, 3, 2, 4, 6, 0, 0, 1;
    check(!validHomography(singular), "invalid: matrix of rank 2");
    check(!validHomography(singular * std::nan("")), "invalid: NaN elements");

    // RANSAC: 5 correct matches of the shift and 2 wrong ones
    m = makeMatches(shift, {{10, 10}, {120, 15}, {60, 90}, {200, 140}, {35, 170}});
    m.push_back({0, 0, 30, 30, 200, 150});
    m.push_back({0, 0, 70, 50, 15, 180});
    RansacParams params;
    std::mt19937 rng(1);
    RansacResult r = ransac(m, params, rng);
    check(r.ok && r.inliers == std::vector<int>({0, 1, 2, 3, 4}) && sameHomography(r.H, shift),
          "RANSAC: finds the 5 correct matches and the shift, rejects the 2 wrong ones");
    check(r.trials < params.Tmax, "RANSAC: the adaptive limit stops before Tmax");

    std::mt19937 rng2(1);
    RansacResult r2 = ransac(m, params, rng2);
    check(r2.ok && r2.inliers == r.inliers && r2.trials == r.trials && (r2.H - r.H).norm() == 0,
          "RANSAC: same seed, same result");

    // RANSAC failures
    m = makeMatches(shift, {{0, 0}, {100, 0}, {0, 80}});
    check(!ransac(m, params, rng).ok, "RANSAC: fails with fewer than 4 matches");
    m = makeMatches(shift, {{0, 0}, {10, 0}, {20, 0}, {30, 0}, {40, 0}, {50, 0}});
    check(!ransac(m, params, rng).ok, "RANSAC: fails when all points are on one line");

    printf("%s\n", failed == 0 ? "all checks passed" : "some checks FAILED");
    return failed == 0 ? 0 : 1;
}
