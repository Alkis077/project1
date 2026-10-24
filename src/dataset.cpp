#include "dataset.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "descriptors.h"

static Image hpImage(const std::string &hpDir, const std::string &seq, int n)
{
    Image img;
    img.id = seq + "/" + std::to_string(n);
    img.path = hpDir + "/" + seq + "/" + std::to_string(n) + ".ppm";
    img.seq = seq;
    return img;
}

Dataset loadDataset(const std::string &hpDir, const std::string &splitFile,
                    const std::string &set)
{
    std::ifstream in(splitFile);
    if (!in)
        throw std::runtime_error("cannot open split file " + splitFile);

    Dataset ds;
    std::string seq, part;
    while (in >> seq >> part) {
        if (part == "train") {
            for (int n = 1; n <= 6; n++)
                ds.train.push_back(hpImage(hpDir, seq, n));
        } else if (part == set) {
            ds.queries.push_back(hpImage(hpDir, seq, 1));
            for (int n = 2; n <= 6; n++)
                ds.base.push_back(hpImage(hpDir, seq, n));
        }
    }
    return ds;
}

void addDistractors(Dataset &ds, const std::string &mirDir, int D,
                    const std::string &cacheDir)
{
    const int total = 25000;
    int added = 0;
    for (int i = 1; i <= total && added < D; i++) {
        Image img;
        img.id = "im" + std::to_string(i) + ".jpg";
        img.path = mirDir + "/" + img.id;

        if (getDescriptors(img, cacheDir).rows == 0) {
            std::cerr << "Skipping " << img.id << " (no SIFT descriptors)\n";
            continue;
        }
        ds.base.push_back(img);
        added++;

        if (added % 1000 == 0)
            std::cerr << "MIRFlickr: " << added << " / " << D << "\n";
    }
    if (added < D)
        std::cerr << "Warning: only " << added << " MIRFlickr images found\n";
}

char category(const Image &img)
{
    if (img.seq.empty())
        return 0;
    return img.seq[0];
}
