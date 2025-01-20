#include "operations.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"

namespace statkitcpp {

Tensor ReshapeOperation::Forward(Tensor& arg, const std::vector<size_t>& shape) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = &arg;
    auto output = ReshapeImpl(arg, shape);
    output.SetRequiresGrad(requires_grad);
    return output;
}

void ReshapeOperation::Backward(const Tensor& grad_output, [[maybe_unused]]const Tensor& output) {
    if (arg_->GetRequiresGrad()) {
        auto darg = ReshapeImpl(grad_output, arg_->GetShape());
        arg_->Backward(darg, output);
    }
}

std::string ReshapeOperation::GetName() const {
    return "ReshapeOperation()";
}
}