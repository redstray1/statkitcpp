#include "Operations.h"
#include "errors.h"
#include "shape.h"
#include <functional>
#include <numeric>

namespace statkitcpp {

Tensor ReshapeImpl(const Tensor &arg, const std::vector<size_t>& shape) {
    size_t new_size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    if (new_size != arg.GetSize()) {
        throw ReshapeError{arg.GetSize(), ShapeToString(shape)};
    }
    Tensor output(arg.GetStorage(), shape, arg.GetDType());
    return output;
}

Tensor UnsqueezeImpl(const Tensor &arg, int dim) {
    auto shape = arg.GetShape();
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    shape.insert(shape.begin() + dim, 1);
    Tensor output(arg.GetStorage(), shape, arg.GetDType());
    return output;
}

}