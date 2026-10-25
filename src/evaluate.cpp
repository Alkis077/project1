#include "evaluate.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Metrics {
    double recall5 = 0, recall10 = 0, ap10 = 0;
};

// result: indices in db.base, closest first. Missing positions (fewer than
// TOP_K results) count as non-relevant.
static Metrics computeMetrics(const std::vector<int> &result, const Database &db,
                              const Image &query)
{
    Metrics m;
    int found = 0;   // relevant images seen so far
    for (int r = 0; r < (int)result.size() && r < TOP_K; r++) {
        if (db.base[result[r]].seq != query.seq)
            continue;
        found++;
        // AP@10 = (1/5) * sum of j / r_j, where the j-th relevant image is at rank r_j
        m.ap10 += (double)found / (r + 1);
        if (r < 5)
            m.recall5 += 1;
        m.recall10 += 1;
    }
    m.recall5 /= RELEVANT;
    m.recall10 /= RELEVANT;
    m.ap10 /= RELEVANT;
    return m;
}

// sums of the metrics over a group of queries
struct Totals {
    Metrics sum;
    double timeMs = 0;
    int count = 0;

    void add(const Metrics &m, double ms)
    {
        sum.recall5 += m.recall5;
        sum.recall10 += m.recall10;
        sum.ap10 += m.ap10;
        timeMs += ms;
        count++;
    }
};

static void printTotals(const std::string &label, const Totals &t)
{
    if (t.count == 0)
        return;
    std::cout << label << " (" << t.count << " queries): "
              << "Recall@5 " << t.sum.recall5 / t.count
              << ", Recall@10 " << t.sum.recall10 / t.count
              << ", mAP@10 " << t.sum.ap10 / t.count
              << ", time " << t.timeMs / t.count << " ms\n";
}

void evaluate(const SearchMethod &method, const Database &db, const std::string &outFile)
{
    std::ofstream out(outFile);
    if (!out)
        throw std::runtime_error("cannot write " + outFile);

    out << method.name() << "\n";

    Totals all, illum, view;
    int shortResults = 0;   // queries with fewer than TOP_K results

    for (size_t q = 0; q < db.queries.size(); q++) {
        // only the search itself is timed
        auto start = std::chrono::steady_clock::now();
        std::vector<int> result = method.search(db.query(q), TOP_K);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        const Image &query = db.queries[q];
        Metrics m = computeMetrics(result, db, query);
        if ((int)result.size() < TOP_K)
            shortResults++;

        out << "Query: " << query.seq << "\n";
        for (int r = 0; r < TOP_K; r++) {
            out << "Rank-" << r + 1 << ": ";
            if (r < (int)result.size())
                out << db.base[result[r]].id;
            else
                out << "-";   // the method returned fewer results
            out << "\n";
        }
        out << "Recall@5: " << m.recall5 << "\n"
            << "Recall@10: " << m.recall10 << "\n"
            << "AP@10: " << m.ap10 << "\n";

        all.add(m, ms);
        if (category(query) == 'i')
            illum.add(m, ms);
        else
            view.add(m, ms);
    }

    int n = all.count;
    out << "Mean Recall@5: " << all.sum.recall5 / n << "\n"
        << "Mean Recall@10: " << all.sum.recall10 / n << "\n"
        << "mAP@10: " << all.sum.ap10 / n << "\n"
        << "Average query time: " << all.timeMs / n << " ms\n";

    std::cout << method.name() << "\n";
    printTotals("All", all);
    printTotals("Illumination", illum);
    printTotals("Viewpoint", view);
    std::cout << "Queries with fewer than " << TOP_K << " results: " << shortResults << "\n";
    std::cout << "Output written to " << outFile << "\n";
}
