
#define TENSOR_BINARY_DECLARATIONS_WITH_OP(_) \
    _(+, Add, AddFunction, add) \
    _(-, Sub, SubFunction, sub) \
    _(*, Mul, MulFunction, mul) \
    _(/, Div, DivFunction, div) \

#define TENSOR_BINARY_DECLARATIONS_WITHOUT_OP(_) \
    _(**, Pow, PowFunction, pow) \

#define TENSOR_AGGREGATION_METHODS(_) \
    _(Sum, SumFunction, sum) \
    _(Prod, ProdFunction, prod) \
    _(Mean, MeanFunction, mean) \
    _(Var, VarFunction, var) \

#define TENSOR_POINTWISE_METHODS(_) \
    _(Exp, ExpFunction, exp) \
    _(Log, LogFunction, log) \
    _(Neg, NegFunction, neg) \
    _(Sqrt, SqrtFunction, sqrt)




#define OPERATOR_METHODS_DECLARATIONS(op, name, func, _) \
    Tensor name(Tensor& other); \
    Tensor name(const Scalar& other); \

#define OPERATOR_DECLARATIONS(op, name, func, _) \
    inline Tensor operator op(Tensor& other); \
    inline Tensor operator op(const Scalar& other);



#define OPERATOR_METHODS_DEFINITIONS(op, name, func, _) \
Tensor Tensor::name(Tensor& other) { \
    auto operation = std::make_shared<func>(); \
    auto result = operation->Forward(*this, other); \
    result.grad_fn = operation; \
    return result; \
} \
\
Tensor Tensor::name(const Scalar& other) { \
    auto operation = std::make_shared<func>(); \
    auto result = operation->Forward(*this, other); \
    result.grad_fn = operation; \
    return result; \
} \

#define OPERATORS_DEFINITIONS(op, name, func, _) \
inline Tensor Tensor::operator op(Tensor& other) { \
    return name(other); \
} \
inline Tensor Tensor::operator op(const Scalar& other) { \
    return name(other); \
}



#define AGGREGATION_DECLARATIONS(name, func, _) \
Tensor name(int dim = -1, bool keepdims = false);

#define AGGREGATION_DEFINITIONS(name, func, _) \
Tensor Tensor::name(int dim, bool keepdims) { \
    auto op = std::make_shared<func>(); \
    auto result = op->Forward(*this, dim, keepdims); \
    result.grad_fn = op; \
    return result; \
}


#define POINTWISE_DECLARATIONS(name, func, _) \
Tensor name();

#define POINTWISE_DEFINITIONS(name, func, _) \
Tensor Tensor::name() { \
    auto op = std::make_shared<func>(); \
    auto result = op->Forward(*this); \
    result.grad_fn = op; \
    return result; \
}
