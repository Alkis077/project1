#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>

struct Image {
    std::string id;     // name used in the output, e.g. "v_bird/2" or "im123.jpg"
    std::string path;   // full path of the file
    std::string seq;    // HPatches sequence, empty for MIRFlickr
};

struct Dataset {
    std::vector<Image> train;    // all 6 images of the training sequences
    std::vector<Image> queries;  // image 1 of each sequence of the chosen set
    std::vector<Image> base;     // images 2..6 of the chosen set + MIRFlickr
};

// builds train / queries / base from the split file.
// set is "validation" or "test"
Dataset loadDataset(const std::string &hpDir, const std::string &splitFile,
                    const std::string &set);

// appends the first D MIRFlickr images (im1.jpg, im2.jpg, ...) to the base.
// Images without any SIFT descriptor are skipped and replaced by the next one.
void addDistractors(Dataset &ds, const std::string &mirDir, int D,
                    const std::string &cacheDir);

// 'i' for illumination, 'v' for viewpoint, 0 for MIRFlickr
char category(const Image &img);

#endif
