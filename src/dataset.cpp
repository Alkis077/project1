#include "dataset.h"

#include <fstream>
#include <stdexcept>

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

Image mirImage(const std::string &mirDir, int i)
{
    Image img;
    img.id = "im" + std::to_string(i) + ".jpg";
    img.path = mirDir + "/" + img.id;
    return img;
}

char category(const Image &img)
{
    if (img.seq.empty())
        return 0;
    return img.seq[0];
}
