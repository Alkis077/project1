#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "bovw.h"
#include "dataset.h"
#include "descriptors.h"
#include "evaluate.h"
#include "groundtruth.h"
#include "matching.h"
#include "methods.h"
#include "ransac.h"
#include "vocabulary.h"

// 1st stage for one query: the C closest base images with the method of the
// 1st assignment (indices in db.base, closest first) and the time it took.
struct Candidates {
    std::vector<int> images;
    double timeMs = 0;
};

static Candidates findCandidates(const SearchMethod &method, const Database &db, int q, int C)
{
    Candidates c;
    auto start = std::chrono::steady_clock::now();
    c.images = method.search(db.query(q), C);
    auto end = std::chrono::steady_clock::now();
    c.timeMs = std::chrono::duration<double, std::milli>(end - start).count();
    return c;
}

// SIFT keypoints and descriptors of the candidates (read from the cache, or
// computed the first time). This is not timed, like the SIFT extraction.
static std::vector<Descriptors> loadCandidates(const Database &db, const std::vector<int> &images,
                                               const std::string &cacheDir)
{
    std::vector<Descriptors> desc(images.size());
    std::string error;
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < (int)images.size(); c++) {
        // an exception must not leave the parallel loop, so it is kept and
        // thrown again after the loop
        try {
            desc[c] = getDescriptors(db.base[images[c]], cacheDir);
        } catch (const std::exception &e) {
            #pragma omp critical
            error = e.what();
        }
    }
    if (!error.empty())
        throw std::runtime_error(error);
    return desc;
}

// 2nd stage result for one candidate
struct Verification {
    int image = 0;            // index in db.base
    int matches = 0;          // M: matches accepted by the Lowe test
    bool ok = false;          // RANSAC found a valid homography
    int inliers = 0;          // final inliers (0 if not ok)
    double medianError = 0;   // median error of the final inliers
    Eigen::Matrix3d H;
};

// SIFT matching and RANSAC between query q and each of its candidates.
// Every pair (q, candidate) gets its own random generator, made from the seed
// and the two images, so the result does not depend on the order in which the
// threads run the candidates.
static std::vector<Verification> verifyCandidates(const Descriptors &query,
                                                  const std::vector<Descriptors> &desc,
                                                  const std::vector<int> &images, int q,
                                                  const Args &args)
{
    RansacParams params;
    params.tau = args.tau;
    params.Tmax = args.Tmax;
    params.p = args.p;
    params.adaptive = args.adaptive;

    std::vector<Verification> ver(images.size());
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < (int)images.size(); c++) {
        Verification &v = ver[c];
        v.image = images[c];

        std::vector<Match> matches = matchSift(query, desc[c], args.rho);
        v.matches = matches.size();

        std::seed_seq seq{args.seed, (unsigned)q, (unsigned)images[c]};
        std::mt19937 rng(seq);
        RansacResult r = ransac(matches, params, rng);
        if (r.ok) {
            v.ok = true;
            v.inliers = r.inliers.size();
            v.medianError = r.medianError;
            v.H = r.H;
        }
    }
    return ver;
}

// New order of the candidates (positions in ver, i.e. in the original ranking):
// more final inliers first; same inliers: smaller median error first; still
// the same: original order. Failed candidates have 0 inliers, so they go after
// all the valid ones and keep their original order between them.
// stable_sort keeps the original order for the remaining ties.
static std::vector<int> rankByInliers(const std::vector<Verification> &ver)
{
    std::vector<int> order(ver.size());
    for (int c = 0; c < (int)ver.size(); c++)
        order[c] = c;

    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const Verification &va = ver[a], &vb = ver[b];
        if (va.inliers != vb.inliers)
            return va.inliers > vb.inliers;
        if (va.ok && vb.ok)
            return va.medianError < vb.medianError;
        return false;
    });
    return order;
}

// Order used only for comparison in the experiments: more matches accepted by
// the Lowe test first (no RANSAC), ties in the original order.
static std::vector<int> rankByMatches(const std::vector<Verification> &ver)
{
    std::vector<int> order(ver.size());
    for (int c = 0; c < (int)ver.size(); c++)
        order[c] = c;

    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return ver[a].matches > ver[b].matches;
    });
    return order;
}


// Homography accuracy of one query, with the HPatches reference homographies
// H_1_j (used only here, for the evaluation). Each of the RELEVANT pairs
// (query, image j of the same sequence) is one of:
//   - a failure: image j is not among the candidates, or RANSAC failed, or a
//     corner cannot be projected. It gets no corner error.
//   - valid: RANSAC gave a homography and the four corners can be projected
//   - accurate: valid, with corner error below ACCURATE_PX
struct Accuracy {
    int pairs = 0;
    int valid = 0;
    int accurate = 0;
    double errorSum = 0;   // corner errors of the valid pairs
};

static Accuracy homographyAccuracy(const std::vector<Verification> &ver, const Database &db,
                                   const Image &query, const std::string &hpDir)
{
    Accuracy acc;
    acc.pairs = RELEVANT;

    int width, height;
    if (!imageSize(query.path, width, height))
        throw std::runtime_error("cannot read image " + query.path);

    for (const Verification &v : ver) {
        const Image &img = db.base[v.image];
        if (img.seq != query.seq || !v.ok)
            continue;

        Eigen::Matrix3d Hgt;
        if (!readGroundTruth(hpDir, img.seq, imageNumber(img), Hgt))
            throw std::runtime_error("cannot read the reference homography of " + img.id);

        double error;
        if (!cornerError(v.H, Hgt, width, height, error))
            continue;
        acc.valid++;
        acc.errorSum += error;
        if (error < ACCURATE_PX)
            acc.accurate++;
    }
    return acc;
}

static void addMetrics(Metrics &sum, const Metrics &m)
{
    sum.recall5 += m.recall5;
    sum.recall10 += m.recall10;
    sum.ap10 += m.ap10;
}

// sums over a group of queries (all, illumination or viewpoint)
struct Summary {
    int queries = 0;
    int fewer = 0;                     // queries with fewer than C candidates
    Metrics before, after, matches;    // sums of the metrics of the three rankings
    double candidatesMs = 0, verificationMs = 0;
    Accuracy acc;

    void add(const Metrics &b, const Metrics &a, const Metrics &m, double candMs, double verMs,
             const Accuracy &x, bool fewerThanC)
    {
        queries++;
        if (fewerThanC)
            fewer++;
        addMetrics(before, b);
        addMetrics(after, a);
        addMetrics(matches, m);
        candidatesMs += candMs;
        verificationMs += verMs;
        acc.pairs += x.pairs;
        acc.valid += x.valid;
        acc.accurate += x.accurate;
        acc.errorSum += x.errorSum;
    }
};

// base images in the given order (positions in ver)
static std::vector<int> imagesInOrder(const std::vector<Verification> &ver,
                                      const std::vector<int> &order)
{
    std::vector<int> images;
    for (int c : order)
        images.push_back(ver[c].image);
    return images;
}

static void writeMetrics(std::ofstream &out, const std::string &prefix, const Metrics &m)
{
    out << prefix << "-Recall@5: " << m.recall5 << "\n"
        << prefix << "-Recall@10: " << m.recall10 << "\n"
        << prefix << "-AP@10: " << m.ap10 << "\n";
}

// the part of the output file for one query
static void writeQuery(std::ofstream &out, const Database &db, const Image &query,
                       const std::vector<Verification> &ver, const std::vector<int> &after,
                       const Metrics &before, const Metrics &afterMetrics)
{
    out << "Query: " << query.seq << "\n"
        << "Candidates: " << ver.size() << "\n";

    // a method may return fewer than TOP_K images: the missing positions get
    // "-" as image, 0 inliers and NA error
    for (int r = 0; r < TOP_K; r++)
        out << "Before-Rank-" << r + 1 << ": "
            << (r < (int)ver.size() ? db.base[ver[r].image].id : "-") << "\n";

    for (int r = 0; r < TOP_K; r++) {
        bool exists = r < (int)ver.size();
        const Verification *v = exists ? &ver[after[r]] : nullptr;
        out << "After-Rank-" << r + 1 << ": " << (exists ? db.base[v->image].id : "-") << "\n"
            << "After-Inliers-" << r + 1 << ": " << (exists ? v->inliers : 0) << "\n"
            << "After-MedianError-" << r + 1 << ": ";
        if (exists && v->ok)
            out << v->medianError << "\n";
        else
            out << "NA\n";
    }

    writeMetrics(out, "Before", before);
    writeMetrics(out, "After", afterMetrics);
}

// the last part of the output file (all queries)
static void writeSummary(std::ofstream &out, const Summary &s)
{
    int n = s.queries;
    out << "MeanBeforeRecall@5: " << s.before.recall5 / n << "\n"
        << "MeanBeforeRecall@10: " << s.before.recall10 / n << "\n"
        << "MeanBeforeMAP@10: " << s.before.ap10 / n << "\n"
        << "MeanAfterRecall@5: " << s.after.recall5 / n << "\n"
        << "MeanAfterRecall@10: " << s.after.recall10 / n << "\n"
        << "MeanAfterMAP@10: " << s.after.ap10 / n << "\n"
        << "AverageCandidateTimeMs: " << s.candidatesMs / n << "\n"
        << "AverageVerificationTimeMs: " << s.verificationMs / n << "\n"
        << "AverageTotalTimeMs: " << (s.candidatesMs + s.verificationMs) / n << "\n"
        << "ValidHomographyRate: " << (double)s.acc.valid / s.acc.pairs << "\n"
        << "AccurateHomographyRate5px: " << (double)s.acc.accurate / s.acc.pairs << "\n"
        << "MeanCornerErrorPx: ";
    if (s.acc.valid > 0)
        out << s.acc.errorSum / s.acc.valid << "\n";
    else
        out << "NA\n";
}

// a few lines on the screen for a group of queries, also with the ranking by
// matches (for the experiments)
static void printSummary(const std::string &label, const Summary &s)
{
    if (s.queries == 0)
        return;
    int n = s.queries;
    std::cout << label << " (" << n << " queries, " << s.acc.pairs << " relevant pairs)\n"
              << "  original: Recall@5 " << s.before.recall5 / n << ", Recall@10 "
              << s.before.recall10 / n << ", mAP@10 " << s.before.ap10 / n << "\n"
              << "  matches:  Recall@5 " << s.matches.recall5 / n << ", Recall@10 "
              << s.matches.recall10 / n << ", mAP@10 " << s.matches.ap10 / n << "\n"
              << "  inliers:  Recall@5 " << s.after.recall5 / n << ", Recall@10 "
              << s.after.recall10 / n << ", mAP@10 " << s.after.ap10 / n << "\n"
              << "  time per query: candidates " << s.candidatesMs / n << " ms, verification "
              << s.verificationMs / n << " ms\n"
              << "  valid homographies " << s.acc.valid << " (" << (double)s.acc.valid / s.acc.pairs
              << "), accurate " << s.acc.accurate << " ("
              << (double)s.acc.accurate / s.acc.pairs << "), mean corner error ";
    if (s.acc.valid > 0)
        std::cout << s.acc.errorSum / s.acc.valid << " px\n";
    else
        std::cout << "NA\n";
}

int main(int argc, char **argv)
{
    Args args;
    if (!parseArgs(argc, argv, args, true)) {
        printUsage(argv[0], true);
        return 1;
    }

    try {
        std::unique_ptr<SearchMethod> method = createMethod(args);

        // same data, vocabulary, BoVW vectors and index as in the 1st assignment
        Dataset ds = loadDataset(args.hpDir, args.splitFile, args.set);
        if (ds.queries.empty()) {
            std::cerr << "No queries found for set " << args.set << "\n";
            return 1;
        }
        std::vector<float> vocab = buildVocabulary(ds.train, args.vocab, args.S,
                                                   args.seed, args.cacheDir);
        Database db = buildDatabase(ds, vocab, args.vocab, args.mirDir, args.D, args.set,
                                    args.cacheDir,
                                    vocabName(ds.train, args.vocab, args.S, args.seed));
        std::cerr << "Queries: " << db.queries.size() << ", base: " << db.base.size() << "\n";
        method->build(db);

        std::ofstream out(args.outFile);
        if (!out)
            throw std::runtime_error("cannot write " + args.outFile);
        out << method->name() << "\n";

        Summary all, illum, view;
        for (size_t q = 0; q < db.queries.size(); q++) {
            const Image &queryImg = db.queries[q];

            // 1st stage (timed)
            Candidates cand = findCandidates(*method, db, q, args.C);

            // reading the descriptors is not timed
            Descriptors query = getDescriptors(queryImg, args.cacheDir);
            std::vector<Descriptors> desc = loadCandidates(db, cand.images, args.cacheDir);

            // 2nd stage (timed): matching, RANSAC and the new ranking
            auto start = std::chrono::steady_clock::now();
            std::vector<Verification> ver = verifyCandidates(query, desc, cand.images, q, args);
            std::vector<int> after = rankByInliers(ver);
            auto end = std::chrono::steady_clock::now();
            double verMs = std::chrono::duration<double, std::milli>(end - start).count();

            // evaluation (not timed)
            std::vector<int> byMatches = rankByMatches(ver);
            Metrics mBefore = computeMetrics(cand.images, db, queryImg);
            Metrics mAfter = computeMetrics(imagesInOrder(ver, after), db, queryImg);
            Metrics mMatches = computeMetrics(imagesInOrder(ver, byMatches), db, queryImg);
            Accuracy acc = homographyAccuracy(ver, db, queryImg, args.hpDir);

            writeQuery(out, db, queryImg, ver, after, mBefore, mAfter);

            bool fewer = (int)cand.images.size() < args.C;
            all.add(mBefore, mAfter, mMatches, cand.timeMs, verMs, acc, fewer);
            if (category(queryImg) == 'i')
                illum.add(mBefore, mAfter, mMatches, cand.timeMs, verMs, acc, fewer);
            else
                view.add(mBefore, mAfter, mMatches, cand.timeMs, verMs, acc, fewer);
            std::cerr << "Query " << q + 1 << "/" << db.queries.size() << " done\r";
        }
        writeSummary(out, all);

        std::cout << "\n" << method->name() << ", C = " << args.C << "\n";
        printSummary("All", all);
        printSummary("Illumination", illum);
        printSummary("Viewpoint", view);
        std::cout << "Queries with fewer than " << args.C << " candidates: " << all.fewer << "\n"
                  << "Output written to " << args.outFile << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
