#include "operations.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"

namespace statkitcpp {

Tensor ReshapeOperation::Forward(const Tensor& arg, const std::vector<size_t>& shape) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    auto output = ReshapeImpl(arg, shape);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> ReshapeOperation::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(1);
    if (arg_->GetRequiresGrad()) {
        auto darg = ReshapeImpl(grad_output, arg_->GetShape());
        input_grads[0] = darg;
    }
    return input_grads;
}

std::string ReshapeOperation::GetName() const {
    return "ReshapeOperation()";
}

std::vector<std::shared_ptr<TensorImpl>> ReshapeOperation::GetChildren() {
    return {arg_};
}


Tensor TransposeOperation::Forward(const Tensor& arg, int dim0, int dim1) {
    bool requires_grad = arg.GetRequiresGrad();
    if (requires_grad) {
        arg_ = arg.GetImpl();
        dim0_ = dim0;
        dim1_ = dim1;
    }
    auto output = TransposeImpl(arg, dim0, dim1);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> TransposeOperation::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(1);
    if (arg_->GetRequiresGrad()) {
        auto darg = TransposeImpl(grad_output, dim0_, dim1_);
        input_grads[0] = darg;
    }
    return input_grads;
}

std::string TransposeOperation::GetName() const {
    return "TransposeOperation()";
}

std::vector<std::shared_ptr<TensorImpl>> TransposeOperation::GetChildren() {
    return {arg_};
}

}