#include "groundtruth.h"

#include <cmath>
#include <fstream>

#include <opencv2/imgcodecs.hpp>

#include "homography.h"

bool readGroundTruth(const std::string &hpDir, const std::string &seq, int j, Eigen::Matrix3d &H)
{
    std::ifstream in(hpDir + "/" + seq + "/H_1_" + std::to_string(j));
    if (!in)
        return false;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            if (!(in >> H(r, c)))
                return false;
    return H.allFinite();
}

int imageNumber(const Image &img)
{
    return std::stoi(img.id.substr(img.id.rfind('/') + 1));
}

bool imageSize(const std::string &path, int &width, int &height)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty())
        return false;
    width = img.cols;
    height = img.rows;
    return true;
}

bool cornerError(const Eigen::Matrix3d &H, const Eigen::Matrix3d &Hgt, int width, int height,
                 double &error)
{
    // corners of the first image, at the centers of the corner pixels
    // (the same pixel coordinates as the SIFT keypoints)
    const double w = width - 1, h = height - 1;
    const double corners[4][2] = {{0, 0}, {w, 0}, {w, h}, {0, h}};

    double sum = 0;
    for (const auto &a : corners) {
        double u, v, ugt, vgt;
        if (!project(H, a[0], a[1], u, v) || !project(Hgt, a[0], a[1], ugt, vgt))
            return false;
        sum += std::hypot(u - ugt, v - vgt);
    }
    error = sum / 4;
    return std::isfinite(error);
}
