#pragma  once

#include "../Tensor.h"
#include "ScalarType.h"

namespace statkitcpp {

//Binary operations
template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Tensor& rhs, BinaryOperation op, ScalarType out_type = ScalarType::Undefined);

template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Scalar& rhs, BinaryOperation op, ScalarType out_type = ScalarType::Undefined);

Tensor AddImpl(const Tensor& lhs, const Tensor& rhs);
Tensor AddImpl(const Tensor& lhs, const Scalar& rhs);

Tensor SubImpl(const Tensor& lhs, const Tensor& rhs);
Tensor SubImpl(const Tensor& lhs, const Scalar& rhs);

Tensor MulImpl(const Tensor& lhs, const Tensor& rhs);
Tensor MulImpl(const Tensor& lhs, const Scalar& rhs);

Tensor DivImpl(const Tensor& lhs, const Tensor& rhs);
Tensor DivImpl(const Tensor& lhs, const Scalar& rhs);

Tensor PowImpl(const Tensor& lhs, const Tensor& rhs);
Tensor PowImpl(const Tensor& lhs, const Scalar& rhs);

Tensor EqualImpl(const Tensor& lhs, const Tensor& rhs);
Tensor EqualImpl(const Tensor& lhs, const Scalar& rhs);

Tensor CloseImpl(const Tensor& lhs, const Tensor& rhs, long double atol);
Tensor CloseImpl(const Tensor& lhs, const Scalar& rhs, long double atol);

//Binary derivation
Tensor DivDerivImpl(const Tensor& lhs, const Tensor& rhs);
Tensor PowDerivImpl(const Tensor& lhs, const Tensor& rhs);
Tensor ExpDerivImpl(const Tensor& lhs, const Tensor& rhs);

//Aggregation operations
Tensor SumImpl(const Tensor& arg, int dim, bool keepdims);
Tensor ProdImpl(const Tensor& arg, int dim, bool keepdims);
Tensor MeanImpl(const Tensor& arg, int dim, bool keepdims);
Tensor VarImpl(const Tensor& arg, int dim, bool keepdims);
Tensor MaxImpl(const Tensor& arg, int dim, bool keepdims);
Tensor MinImpl(const Tensor& arg, int dim, bool keepdims);

//Aggregation derivative
Tensor MaxDerivImpl(const Tensor& arg, int dim);
Tensor MinDerivImpl(const Tensor& arg, int dim);

//Pointwise operations
Tensor NegImpl(const Tensor& arg);
Tensor ExpImpl(const Tensor& arg);
Tensor LogImpl(const Tensor& arg);
Tensor SqrtImpl(const Tensor& arg);
Tensor ReciprocalImpl(const Tensor& arg);

//Pointwise derivatives
Tensor SqrtDerivImpl(const Tensor& arg);

//Tensor operations
Tensor ReshapeImpl(const Tensor& arg, const std::vector<size_t>& shape);
Tensor UnsqueezeImpl(const Tensor& arg, int dim);
Tensor SqueezeImpl(const Tensor& arg, int dim);
}