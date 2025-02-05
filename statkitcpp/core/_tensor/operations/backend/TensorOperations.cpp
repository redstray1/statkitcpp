#include "Operations.h"
#include "ScalarType.h"
#include "Tensor.h"
#include "errors.h"
#include "shape.h"
#include "views_vectorized.h"
#include <functional>
#include <numeric>
#include <stdexcept>

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

Tensor SqueezeImpl(const Tensor &arg, int dim) {
    auto shape = arg.GetShape();
    if (dim < 0) {
        dim += arg.GetNDim();
    }
    if (shape[dim] == 1) {
        shape.erase(shape.begin() + dim);
    }
    Tensor output(arg.GetStorage(), shape, arg.GetDType());
    return output;
}

Tensor TransposeImpl(const Tensor& arg, int dim0, int dim1) {
    auto shape = arg.GetShape();
    if (dim0 < 0) {
        dim0 += arg.GetNDim();
    }
    if (dim1 < 0) {
        dim1 += arg.GetNDim();
    }
    std::swap(shape[dim0], shape[dim1]);
    Tensor output(shape, arg.GetDType());
    if (dim0 == static_cast<int>(arg.GetNDim()) - 2 && dim1 == static_cast<int>(arg.GetNDim()) - 1) {
        #define DEFINE_TYPE(T, name) \
        case (ScalarType::name): \
            vec::batched_transpose(arg.data_ptr<T>(), output.data_ptr<T>(), arg.GetDim(-2), arg.GetDim(-1), output.GetSize()); \
            break;
        switch (output.GetDType()) {
            SCALAR_TYPES(DEFINE_TYPE)
            default:
                throw InvalidDatatypeError{};
        }
        #undef DEFINE_TYPE
        return output;
    }
    throw std::runtime_error{"Transpose for not last two dimensions is not implemented yet."};
}

}