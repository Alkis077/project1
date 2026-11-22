#include "ransac.h"

#include <algorithm>
#include <cmath>

#include "homography.h"

// inliers of H among all matches: error < tau. Also returns their median error.
static std::vector<int> findInliers(const std::vector<Match> &matches, const Eigen::Matrix3d &H,
                                   double tau, double &median)
{
    std::vector<int> inliers;
    std::vector<double> errors;
    for (int i = 0; i < (int)matches.size(); i++) {
        double e = reprojectionError(H, matches[i]);
        if (e < tau) {
            inliers.push_back(i);
            errors.push_back(e);
        }
    }

    median = 0;
    int n = errors.size();
    if (n > 0) {
        std::sort(errors.begin(), errors.end());
        median = (n % 2 == 1) ? errors[n / 2] : (errors[n / 2 - 1] + errors[n / 2]) / 2;
    }
    return inliers;
}

// 4 different random matches
static void randomSample(int M, std::mt19937 &rng, int idx[4])
{
    std::uniform_int_distribution<int> pick(0, M - 1);
    for (int k = 0; k < 4; k++) {
        bool repeated;
        do {
            idx[k] = pick(rng);
            repeated = false;
            for (int j = 0; j < k; j++)
                if (idx[j] == idx[k])
                    repeated = true;
        } while (repeated);
    }
}

// Number of trials needed for probability p of one clean sample when the
// inlier ratio is w: ceil(log(1 - p) / log(1 - w^4)).
// log1p(-w^4) is used instead of log(1 - w^4), because for small w the
// subtraction 1 - w^4 loses accuracy.
static double neededTrials(double w, double p)
{
    return std::ceil(std::log(1 - p) / std::log1p(-std::pow(w, 4)));
}

RansacResult ransac(const std::vector<Match> &matches, const RansacParams &params,
                    std::mt19937 &rng)
{
    RansacResult res;
    int M = matches.size();
    if (M < 4)
        return res;

    bool found = false;
    std::vector<int> best;
    double bestMedian = 0;
    double limit = params.Tmax;

    for (int t = 1; t <= params.Tmax; t++) {
        if (t > limit)
            break;
        res.trials = t;

        int idx[4];
        randomSample(M, rng, idx);
        if (degenerateSample(matches, idx))
            continue;

        Eigen::Matrix3d H;
        if (!estimateHomography(matches, std::vector<int>(idx, idx + 4), H))
            continue;

        double median;
        std::vector<int> inliers = findInliers(matches, H, params.tau, median);

        // more inliers, or as many with a smaller median error
        bool better = !found || inliers.size() > best.size() ||
                      (inliers.size() == best.size() && median < bestMedian);
        if (!better)
            continue;

        found = true;
        best = inliers;
        bestMedian = median;

        if (params.adaptive) {
            double w = (double)best.size() / M;
            if (w >= 1)
                break;   // every match is an inlier: no reason to keep trying
            if (w > 0)   // w = 0 would divide by log(1) = 0, so the limit stays
                limit = std::min(limit, neededTrials(w, params.p));
        }
    }

    if (!found || best.size() < 4)
        return res;

    // refit with the inliers of the best hypothesis only
    // (estimateHomography computes new T, T' from these points)
    Eigen::Matrix3d H;
    if (!estimateHomography(matches, best, H))
        return res;

    // final inliers among all M matches
    res.inliers = findInliers(matches, H, params.tau, res.medianError);
    if (res.inliers.size() < 4)
        return res;

    res.H = H;
    res.ok = true;
    return res;
}
