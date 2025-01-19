#pragma  once

#include "../Tensor.h"

namespace statkitcpp {

template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Tensor& rhs, BinaryOperation op);

Tensor AddImpl(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha);
Tensor SubImpl(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha);
Tensor MulImpl(const Tensor& lhs, const Tensor& rhs);
Tensor DivImpl(const Tensor& lhs, const Tensor& rhs);

Tensor SumImpl(const Tensor& arg, int dim, bool keepdims);
Tensor MeanImpl(const Tensor& arg, int dim, bool keepdims);
Tensor VarImpl(const Tensor& arg, int dim, bool keepdims);
}