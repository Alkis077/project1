#include "homography.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/SVD>

// Normalization of one point set: T moves the centroid to the origin and
// scales so that the mean distance from the origin is sqrt(2).
//     | s  0  -s*cx |
// T = | 0  s  -s*cy |      s = sqrt(2) / mean distance
//     | 0  0    1   |
// Returns false if the mean distance is almost zero (all points the same).
static bool normalization(const std::vector<double> &px, const std::vector<double> &py,
                          Eigen::Matrix3d &T, double &s, double &cx, double &cy)
{
    int n = px.size();
    cx = cy = 0;
    for (int i = 0; i < n; i++) {
        cx += px[i];
        cy += py[i];
    }
    cx /= n;
    cy /= n;

    double mean = 0;
    for (int i = 0; i < n; i++)
        mean += std::hypot(px[i] - cx, py[i] - cy);
    mean /= n;
    if (mean < 1e-9)
        return false;

    s = std::sqrt(2.0) / mean;
    T << s, 0, -s * cx,
         0, s, -s * cy,
         0, 0, 1;
    return true;
}

bool estimateHomography(const std::vector<Match> &matches, const std::vector<int> &idx,
                        Eigen::Matrix3d &H)
{
    int n = idx.size();
    if (n < 4)
        return false;

    std::vector<double> x(n), y(n), u(n), v(n);
    for (int k = 0; k < n; k++) {
        const Match &m = matches[idx[k]];
        x[k] = m.x;
        y[k] = m.y;
        u[k] = m.u;
        v[k] = m.v;
    }

    // T for the first image, T' for the second
    Eigen::Matrix3d T, T2;
    double s, cx, cy, s2, cx2, cy2;
    if (!normalization(x, y, T, s, cx, cy) || !normalization(u, v, T2, s2, cx2, cy2))
        return false;

    // two rows of A for every match, with the normalized coordinates
    Eigen::MatrixXd A(2 * n, 9);
    for (int k = 0; k < n; k++) {
        double xn = s * (x[k] - cx), yn = s * (y[k] - cy);
        double un = s2 * (u[k] - cx2), vn = s2 * (v[k] - cy2);
        A.row(2 * k) << -xn, -yn, -1, 0, 0, 0, un * xn, un * yn, un;
        A.row(2 * k + 1) << 0, 0, 0, -xn, -yn, -1, vn * xn, vn * yn, vn;
    }

    // h = right singular vector of the smallest singular value = last column
    // of V. With 4 matches A is 8x9, so the full V is needed to get it.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(8);

    Eigen::Matrix3d Hn;
    Hn << h(0), h(1), h(2),
          h(3), h(4), h(5),
          h(6), h(7), h(8);

    // back to pixel coordinates: H = T'^-1 Hn T (T' is inverted directly)
    Eigen::Matrix3d T2inv;
    T2inv << 1 / s2, 0, cx2,
             0, 1 / s2, cy2,
             0, 0, 1;
    H = T2inv * Hn * T;

    // H is defined up to scale, so it is scaled to norm 1
    double norm = H.norm();
    if (!std::isfinite(norm) || norm == 0)
        return false;
    H /= norm;
    return validHomography(H);
}

// eta = |(b - a) x (c - a)| / L^2, L = longest side of the triangle.
// The triple is (almost) collinear if L = 0 or eta < EPS_COL.
static bool collinear(double ax, double ay, double bx, double by, double cx, double cy)
{
    double L = std::max({std::hypot(bx - ax, by - ay), std::hypot(cx - ax, cy - ay),
                         std::hypot(cx - bx, cy - by)});
    if (L == 0)
        return true;
    double cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    return std::fabs(cross) / (L * L) < EPS_COL;
}

// checks the 4 points of one image (useSecond: points (u, v) instead of (x, y))
static bool degeneratePoints(const std::vector<Match> &matches, const int idx[4], bool useSecond)
{
    double px[4], py[4];
    for (int k = 0; k < 4; k++) {
        const Match &m = matches[idx[k]];
        px[k] = useSecond ? m.u : m.x;
        py[k] = useSecond ? m.v : m.y;
    }

    // repeated points
    for (int a = 0; a < 4; a++)
        for (int b = a + 1; b < 4; b++)
            if (px[a] == px[b] && py[a] == py[b])
                return true;

    // every triple of the 4 points
    for (int a = 0; a < 4; a++)
        for (int b = a + 1; b < 4; b++)
            for (int c = b + 1; c < 4; c++)
                if (collinear(px[a], py[a], px[b], py[b], px[c], py[c]))
                    return true;
    return false;
}

bool degenerateSample(const std::vector<Match> &matches, const int idx[4])
{
    return degeneratePoints(matches, idx, false) || degeneratePoints(matches, idx, true);
}

bool validHomography(const Eigen::Matrix3d &H)
{
    if (!H.allFinite())
        return false;

    // singular values s1 >= s2 >= s3; the ratio does not depend on the scale of H
    Eigen::MatrixXd M = H;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);
    Eigen::VectorXd sv = svd.singularValues();
    return sv(0) > 0 && sv(2) / sv(0) > EPS_RANK;
}

bool project(const Eigen::Matrix3d &H, double x, double y, double &u, double &v)
{
    Eigen::Vector3d p(x, y, 1);
    Eigen::Vector3d q = H * p;
    if (std::fabs(q(2)) <= EPS_PROJ * H.row(2).norm() * p.norm())
        return false;

    u = q(0) / q(2);
    v = q(1) / q(2);
    return std::isfinite(u) && std::isfinite(v);
}

double reprojectionError(const Eigen::Matrix3d &H, const Match &m)
{
    double u, v;
    if (!project(H, m.x, m.y, u, v))
        return std::numeric_limits<double>::infinity();
    return std::hypot(m.u - u, m.v - v);
}
