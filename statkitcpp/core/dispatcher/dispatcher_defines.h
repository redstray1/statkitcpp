#define DISPATCHER_OPERATOR_METHODS_DECLARATIONS(op, name, func, _) \
    TensorDispatcher name(const TensorDispatcher& other) const; \
    TensorDispatcher name(const Scalar& other) const; \

#define DISPATCHER_OPERATOR_DECLARATIONS(op, name, func, _) \
    inline TensorDispatcher operator op(const TensorDispatcher& other) const; \
    inline TensorDispatcher operator op(const Scalar& other) const;

#define DISPATCHER_AGGREGATION_DECLARATIONS(name, func, _) \
TensorDispatcher name(int dim, bool keepdims) const;

#define DISPATCHER_POINTWISE_DECLARATIONS(name, func, _) \
TensorDispatcher name() const; 



#define DISPATCHER_AGGREGATION_DEFINITIONS(name, func, _) \
TensorDispatcher TensorDispatcher::name(int dim, bool keepdims) const { \
    auto var_ptr = tensor_.name(dim, keepdims); \
    return TensorDispatcher(var_ptr); \
}

#define DISPATCHER_OPERATOR_METHODS_DEFINITIONS(op, name, func, _) \
TensorDispatcher TensorDispatcher::name(const TensorDispatcher& other) const { \
    auto var_ptr = tensor_.name(other.tensor_); \
    return TensorDispatcher(var_ptr); \
}\
TensorDispatcher TensorDispatcher::name(const Scalar& other) const { \
    return TensorDispatcher(tensor_.name(other)); \
}

#define DISPATCHER_OPERATOR_DEFINITIONS(op, name, func, _) \
inline TensorDispatcher TensorDispatcher::operator op(const TensorDispatcher& other) const { \
    return name(other); \
}\
inline TensorDispatcher TensorDispatcher::operator op(const Scalar& other) const { \
    return name(other); \
}


#define DISPATCHER_POINTWISE_DEFINITIONS(name, func, _) \
TensorDispatcher TensorDispatcher::name() const { \
    return TensorDispatcher(tensor_.name()); \
}