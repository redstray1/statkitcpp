#include <vector>
#include <math.h>
#include "config.h"

class Tensor {
private:
    std::vector<size_t> shape_;
    data_type dtype_;
    std::vector<float> data_;
};