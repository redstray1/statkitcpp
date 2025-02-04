#pragma once

#define TENSOR_BINARY_DECLARATIONS_WITH_OP(_) \
    _(+, Add, AddFunction, add) \
    _(-, Sub, SubFunction, sub) \
    _(*, Mul, MulFunction, mul) \
    _(/, Div, DivFunction, div)

#define TENSOR_BINARY_DECLARATIONS_WITHOUT_OP(_) \
    _(**, Pow, PowFunction, pow)

#define TENSOR_LINALG_OPERATIONS_WITH_OP(_) \
    _(@, MatMul, MatMulFunction, matmul)

#define TENSOR_LINALG_OPERATIONS_WITHOUT_OP(_) \
    _(_, Dot, DotFunction, dot)

#define TENSOR_AGGREGATION_METHODS(_) \
    _(Sum, SumFunction, sum) \
    _(Prod, ProdFunction, prod) \
    _(Mean, MeanFunction, mean) \
    _(Var, VarFunction, var) \
    _(Max, MaxFunction, max) \
    _(Min, MinFunction, min)

#define TENSOR_POINTWISE_METHODS(_) \
    _(Exp, ExpFunction, exp) \
    _(Log, LogFunction, log) \
    _(Neg, NegFunction, neg) \
    _(Sqrt, SqrtFunction, sqrt)




#define OPERATOR_METHODS_DECLARATIONS(op, name, func, _) \
    Tensor name(const Tensor& other) const; \
    Tensor name(const Scalar& other) const;

#define OPERATOR_DECLARATIONS(op, name, func, _) \
    inline Tensor operator op(const Tensor& other) const; \
    inline Tensor operator op(const Scalar& other) const;



#define OPERATOR_METHODS_DEFINITIONS(op, name, func, _) \
Tensor Tensor::name(const Tensor& other) const { \
    auto operation = std::make_shared<func>(); \
    auto result = operation->Forward(*this, other); \
    result.impl_->grad_fn = operation; \
    return result; \
} \
\
Tensor Tensor::name(const Scalar& other) const { \
    auto operation = std::make_shared<func>(); \
    auto result = operation->Forward(*this, other); \
    result.impl_->grad_fn = operation; \
    return result; \
}

#define OPERATORS_DEFINITIONS(op, name, func, _) \
inline Tensor operator op(const Tensor& other) const { \
    return name(other); \
} \
inline Tensor operator op(const Scalar& other) const { \
    return name(other); \
}



#define AGGREGATION_DECLARATIONS(name, func, _) \
Tensor name(int dim = -1, bool keepdims = false) const;

#define AGGREGATION_DEFINITIONS(name, func, _) \
Tensor Tensor::name(int dim, bool keepdims) const { \
    auto op = std::make_shared<func>(); \
    auto result = op->Forward(*this, dim, keepdims); \
    result.impl_->grad_fn = op; \
    return result; \
}


#define POINTWISE_DECLARATIONS(name, func, _) \
Tensor name() const;

#define POINTWISE_DEFINITIONS(name, func, _) \
Tensor Tensor::name() const { \
    auto op = std::make_shared<func>(); \
    auto result = op->Forward(*this); \
    result.impl_->grad_fn = op; \
    return result; \
}



#define LINALG_METHODS_DECLARATIONS(op, name, func, _) \
    Tensor name(const Tensor& other) const;

#define LINALG_METHODS_DEFINITIONS(op, name, func, _) \
Tensor Tensor::name(const Tensor& other) const { \
    auto operation = std::make_shared<func>(); \
    auto result = operation->Forward(*this, other); \
    result.impl_->grad_fn = operation; \
    return result; \
}
