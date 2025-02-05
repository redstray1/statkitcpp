
#include "Operations.h"
#include "TensorIndex.h"
#include "operations.h"
#include "TensorCreationOps.h"
#include "Tensor.h"

namespace statkitcpp {

Tensor IndexingOperation::Forward(const Tensor& arg, const std::vector<TensorIndex>& indices) {
    bool requires_grad = arg.GetRequiresGrad();
    if (requires_grad) {
        arg_ = arg.GetImpl();
        indices_ = indices;
    }
    auto output = IndexingImpl(arg, indices);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> IndexingOperation::Backward(const Tensor& grad_output) {
    std::vector<Tensor> input_grads(1);
    if (arg_->GetRequiresGrad()) {
        auto darg = Zeros(arg_->GetShape(), arg_->GetDType());
        IndexingPutImpl(darg, indices_, grad_output);
        input_grads[0] = darg;
    }
    return input_grads;
}

std::string IndexingOperation::GetName() const {
    return "IndexingOperation()";
}

std::vector<std::shared_ptr<TensorImpl>> IndexingOperation::GetChildren() {
    return {arg_};
}

}