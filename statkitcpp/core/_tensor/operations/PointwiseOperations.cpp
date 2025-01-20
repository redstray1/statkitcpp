#include "Operations.h"
#include "memory_ops.h"
#include "function_objects.h"

namespace statkitcpp {

Tensor NegImpl(const Tensor& arg) {
    Tensor output(arg.GetShape(), arg.GetDType());
    ops::pointwise(arg.GetDataPointer(), output.GetSize(), output.GetDType(), std::negate(), output.GetDataPointer());
    return output;
}

Tensor ExpImpl(const Tensor& arg) {
    Tensor output(arg.GetShape(), arg.GetDType());
    ops::pointwise(arg.GetDataPointer(), output.GetSize(), output.GetDType(), func::exp(), output.GetDataPointer());
    return output;
}

Tensor LogImpl(const Tensor& arg) {
    Tensor output(arg.GetShape(), arg.GetDType());
    ops::pointwise(arg.GetDataPointer(), output.GetSize(), output.GetDType(), func::log(), output.GetDataPointer());
    return output;
}

Tensor SqrtImpl(const Tensor& arg) {
    Tensor output(arg.GetShape(), arg.GetDType());
    ops::pointwise(arg.GetDataPointer(), output.GetSize(), output.GetDType(), func::sqrt(), output.GetDataPointer());
    return output;
}

}