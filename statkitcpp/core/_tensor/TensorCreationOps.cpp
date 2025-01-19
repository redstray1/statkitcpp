#include "TensorCreationOps.h"
#include "ScalarType.h"
#include "Scalar.h"
#include "errors.h"
#include "tensor_creation_ops.h"

namespace statkitcpp {


Tensor Arange(const Scalar& start, const Scalar& end, const Scalar& step, ScalarType dtype) {
    size_t numel = 0;
    #define DEFINE_DTYPE(T, name) \
    case (ScalarType::name): { \
        SKPP_CHECK(start, dtype); \
        SKPP_CHECK(end, dtype); \
        SKPP_CHECK(step, dtype); \
        T start_ = start.to##name(); \
        T end_ = end.to##name(); \
        T step_ = step.to##name(); \
        numel = ceil((end_ - start_) / step_); \
        break; \
    }
    switch(dtype) {
        SCALAR_TYPES(DEFINE_DTYPE)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_DTYPE
    Tensor result({numel}, dtype);
    ops::arange(result, start, end, step);
    return result;
}

Tensor Empty(const std::vector<size_t>& shape, ScalarType dtype);

Tensor Eye(const Scalar& n, const Scalar& m, ScalarType dtype);
Tensor Full(const std::vector<size_t>& shape, const Scalar& fill_value, ScalarType dtype) {
    SKPP_CHECK(fill_value, dtype);
    Tensor result(shape, dtype);
    ops::full(result, fill_value);
    return result;
}
Tensor Ones(const std::vector<size_t>& shape, ScalarType dtype) {
    Tensor result(shape, dtype);
    ops::ones(result);
    return result;
}
Tensor Zeros(const std::vector<size_t>& shape, ScalarType dtype) {
    Tensor result(shape, dtype);
    ops::zeros(result);
    return result;
}




}