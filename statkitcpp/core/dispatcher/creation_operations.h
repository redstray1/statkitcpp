#pragma once

#include "Scalar.h"
#include "ScalarType.h"
#include "datatypes.h"
#include "tensor_dispatcher.h"
#include "../_tensor/TensorCreationOps.h"

namespace statkitcpp {

TensorDispatcher ArangePy(const Scalar& start, const Scalar& end, const Scalar& step = 1, ScalarType dtype = kFloat32) {
    return TensorDispatcher(Arange(start, end, step, dtype));
}

TensorDispatcher FullPy(const std::vector<size_t>& shape, const Scalar& fill_value, ScalarType dtype = kFloat32) {
    return TensorDispatcher(Full(shape, fill_value, dtype));
}

TensorDispatcher OnesPy(const std::vector<size_t>& shape, ScalarType dtype = kFloat32) {
    return TensorDispatcher(Ones(shape, dtype));
}

TensorDispatcher ZerosPy(const std::vector<size_t>& shape, ScalarType dtype = kFloat32) {
    return TensorDispatcher(Zeros(shape, dtype));
}
}