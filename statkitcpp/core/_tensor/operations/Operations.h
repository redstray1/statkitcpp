#pragma  once

#include "../Tensor.h"
#include "ScalarType.h"
#include "config.h"
#include "TensorIndex.h"

namespace statkitcpp {

//Specified binary operations
Tensor AddOptimized(const Tensor& lhs, const Tensor& rhs);
Tensor SubOptimized(const Tensor& lhs, const Tensor& rhs);
Tensor MulOptimized(const Tensor& lhs, const Tensor& rhs);
Tensor DivOptimized(const Tensor& lhs, const Tensor& rhs);

//Binary operations
template <typename BinaryOperation>
Tensor ApplyBinaryOp(const Tensor& lhs, const Tensor& rhs, const BinaryOperation& op, ScalarType out_type = ScalarType::Undefined);

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

Tensor CloseImpl(const Tensor& lhs, const Tensor& rhs, long double atol = kMachinePrecision);
Tensor CloseImpl(const Tensor& lhs, const Scalar& rhs, long double atol = kMachinePrecision);

//Matrix multiplication operation
Tensor DotImpl(const Tensor& lhs, const Tensor& rhs);
Tensor VecMatImpl(const Tensor& lhs, const Tensor& rhs, bool rhs_transposed = false);
Tensor MatVecImpl(const Tensor& lhs, const Tensor& rhs);
Tensor MatMatMulImpl(const Tensor& lhs, const Tensor& rhs, bool rhs_transposed = false);
Tensor BatchedMatMatMulImpl(const Tensor& lhs, const Tensor& rhs, bool rhs_transposed = false);
Tensor MatMulImpl(const Tensor& lhs, const Tensor& rhs, bool rhs_transposed = false);

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
Tensor TransposeImpl(const Tensor& arg, int dim0 = -2, int dim1 = -1);

//Indexing operations
Tensor IndexingImpl(const Tensor& arg, const std::vector<TensorIndex>& indices);
Tensor& IndexingPutImpl(Tensor& arg, const std::vector<TensorIndex>& indices, const Tensor& other);
Tensor& IndexingPutImpl(Tensor& arg, const std::vector<TensorIndex>& indices, const Scalar& other);
}