#include "Operations.h"
#include "errors.h"
#include "shape.h"
#include <functional>
#include <numeric>

namespace statkitcpp {

Tensor ReshapeImpl(const Tensor &arg, const std::vector<size_t>& shape) {
    size_t new_size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    if (new_size != arg.GetSize()) {
        throw ReshapeError{arg.GetSize(), ShapeToString(arg.GetShape())};
    }
    Tensor output(arg.GetStorage(), shape, arg.GetDType());
    return output;
}

}