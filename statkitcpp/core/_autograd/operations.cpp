#include "operations.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"
#include <memory>

namespace statkitcpp {

Tensor AddFunction::Forward(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = std::make_shared<Tensor>(lhs);
    rhs_ = std::make_shared<Tensor>(rhs);
    auto output = AddImpl(lhs, rhs, alpha);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor AddFunction::Backward(const Tensor& grad_output, const Tensor& output) {
    return grad_output;
}

Tensor SubFunction::Forward(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = std::make_shared<Tensor>(lhs);
    rhs_ = std::make_shared<Tensor>(rhs);
    auto output = SubImpl(lhs, rhs, alpha);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor SubFunction::Backward(const Tensor& grad_output, const Tensor& output) {
    return grad_output;
}

Tensor MulFunction::Forward(const Tensor& lhs, const Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = std::make_shared<Tensor>(lhs);
    rhs_ = std::make_shared<Tensor>(rhs);
    auto output = MulImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor MulFunction::Backward(const Tensor& grad_output, const Tensor& output) {
    return grad_output;
}

Tensor DivFunction::Forward(const Tensor& lhs, const Tensor& rhs) {
    bool requires_grad = lhs.GetRequiresGrad() || rhs.GetRequiresGrad();
    lhs_ = std::make_shared<Tensor>(lhs);
    rhs_ = std::make_shared<Tensor>(rhs);
    auto output = DivImpl(lhs, rhs);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor DivFunction::Backward(const Tensor& grad_output, const Tensor& output) {
    return grad_output;
}

Tensor SumFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = std::make_shared<Tensor>(arg);
    auto output = SumImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor SumFunction::Backward(const Tensor& grad_output, const Tensor& output) {
    return grad_output;
}

Tensor MeanFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = std::make_shared<Tensor>(arg);
    auto output = MeanImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor MeanFunction::Backward(const Tensor& grad_output, const Tensor& output) {
    return grad_output;
}

Tensor VarFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = std::make_shared<Tensor>(arg);
    auto output = VarImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

Tensor VarFunction::Backward(const Tensor& grad_output, const Tensor& output) {
    return grad_output;
}

}