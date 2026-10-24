#include "descriptors.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

static std::string cacheFile(const Image &img, const std::string &cacheDir)
{
    if (img.seq.empty())
        return cacheDir + "/mirflickr/" + img.id + ".sift";
    return cacheDir + "/hpatches/" + img.id + ".sift";
}

// OpenCV is used only here: to read the image and to run SIFT.
// The result is copied into our own struct right away.
static Descriptors computeSift(const std::string &path)
{
    cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (gray.empty())
        throw std::runtime_error("cannot read image " + path);

    // one SIFT object per thread, because images may be processed in parallel
    thread_local cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat mat;
    sift->detectAndCompute(gray, cv::noArray(), keypoints, mat);

    Descriptors desc;
    desc.rows = mat.rows;
    desc.data.resize((size_t)mat.rows * DIM);
    for (int i = 0; i < mat.rows; i++) {
        const float *src = mat.ptr<float>(i);
        for (int j = 0; j < DIM; j++)
            desc.data[(size_t)i * DIM + j] = src[j];
    }
    return desc;
}

// file format: number of rows (int), then rows*128 bytes.
// SIFT values are whole numbers in [0,255] so one byte per value is enough.
static void save(const std::string &file, const Descriptors &desc)
{
    fs::create_directories(fs::path(file).parent_path());

    std::vector<uint8_t> bytes(desc.data.size());
    for (size_t i = 0; i < bytes.size(); i++)
        bytes[i] = (uint8_t)std::lround(desc.data[i]);

    // write to a temp file first, so an interrupted run does not leave a broken file
    std::string tmp = file + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char *>(&desc.rows), sizeof(desc.rows));
        out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        if (!out)
            throw std::runtime_error("cannot write " + tmp);
    }
    fs::rename(tmp, file);
}

static bool load(const std::string &file, Descriptors &desc)
{
    std::ifstream in(file, std::ios::binary);
    if (!in)
        return false;

    int rows = 0;
    if (!in.read(reinterpret_cast<char *>(&rows), sizeof(rows)) || rows < 0)
        return false;

    std::vector<uint8_t> bytes((size_t)rows * DIM);
    if (!in.read(reinterpret_cast<char *>(bytes.data()), bytes.size()))
        return false;

    desc.rows = rows;
    desc.data.assign(bytes.begin(), bytes.end());
    return true;
}

Descriptors getDescriptors(const Image &img, const std::string &cacheDir)
{
    if (cacheDir.empty())
        return computeSift(img.path);

    std::string file = cacheFile(img, cacheDir);
    Descriptors desc;
    if (load(file, desc))
        return desc;

    desc = computeSift(img.path);
    save(file, desc);
    return desc;
}
