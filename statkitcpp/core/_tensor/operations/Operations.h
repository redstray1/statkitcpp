#pragma  once

#include "../Tensor.h"

namespace statkitcpp {

//Binary operations
template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Tensor& rhs, BinaryOperation op);

Tensor AddImpl(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha);
Tensor SubImpl(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha);
Tensor MulImpl(const Tensor& lhs, const Tensor& rhs);
Tensor DivImpl(const Tensor& lhs, const Tensor& rhs);
Tensor PowImpl(const Tensor& lhs, const Tensor& rhs);

//Aggregation operations
Tensor SumImpl(const Tensor& arg, int dim, bool keepdims);
Tensor ProdImpl(const Tensor& arg, int dim, bool keepdims);
Tensor MeanImpl(const Tensor& arg, int dim, bool keepdims);
Tensor VarImpl(const Tensor& arg, int dim, bool keepdims);
Tensor MaxImpl(const Tensor& arg, int dim, bool keepdims);
Tensor MinImpl(const Tensor& arg, int dim, bool keepdims);

//Pointwise operations
Tensor NegImpl(const Tensor& arg);
Tensor ExpImpl(const Tensor& arg);
Tensor LogImpl(const Tensor& arg);
Tensor SqrtImpl(const Tensor& arg);

//Tensor operations
Tensor ReshapeImpl(const Tensor& arg, const std::vector<size_t>& shape);
}