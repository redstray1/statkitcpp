#pragma once

#include "ScalarType.h"
#include "datatypes.h"
#include "Tensor.h"

namespace statkitcpp {

Tensor Arange(const Scalar& start, const Scalar& end, const Scalar& step=1, ScalarType dtype = kFloat32);

Tensor Empty(const std::vector<size_t>& shape, ScalarType dtype = kFloat32);

Tensor Eye(const Scalar& n, const Scalar& m, ScalarType dtype = kFloat32);

Tensor Full(const std::vector<size_t>& shape, const Scalar& fill_value, ScalarType dtype = kFloat32);
Tensor Ones(const std::vector<size_t>& shape, ScalarType dtype = kFloat32);
Tensor Zeros(const std::vector<size_t>& shape, ScalarType dtype = kFloat32);



}