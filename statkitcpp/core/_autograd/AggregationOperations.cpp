#include "operations.h"
#include "tensor_creation_ops.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"

namespace statkitcpp {


Tensor SumFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    auto output = SumImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> SumFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg(arg_->GetShape(), arg_->GetDType());
        ops::ones(darg);
        darg = MulImpl(darg, grad_output);
        dargs[0] = darg;
    }
    return dargs;
}

std::string SumFunction::GetName() const {
    return "SumFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> SumFunction::GetChildren() {
    return {arg_};
}



Tensor ProdFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    auto output = ProdImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> ProdFunction::Backward(const Tensor& grad_output) {
    return {grad_output};
}

std::string ProdFunction::GetName() const {
    return "ProdFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> ProdFunction::GetChildren() {
    return {arg_};
}




Tensor MeanFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    dim_ = dim;
    auto output = MeanImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}



std::vector<Tensor> MeanFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg(arg_->GetShape(), arg_->GetDType());
        ops::ones(darg);
        darg = MulImpl(darg, grad_output);
        darg = DivImpl(darg, static_cast<int>(arg_->GetDim(dim_)));
        dargs[0] = darg;
    }
    return dargs;
}

std::string MeanFunction::GetName() const {
    return "MeanFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> MeanFunction::GetChildren() {
    return {arg_};
}





Tensor VarFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    dim_ = dim;
    auto output = VarImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> VarFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg(arg_->GetShape(), arg_->GetDType());
        ops::ones(darg);
        Tensor temp(arg_);
        darg = MulImpl(darg, grad_output);
        darg = MulImpl(darg, 2);
        darg = MulImpl(darg, SubImpl(temp, MeanImpl(temp, dim_, true)));
        darg = DivImpl(darg, static_cast<int>(arg_->GetShape()[dim_]));
        dargs[0] = darg;
    }
    return dargs;
}

std::string VarFunction::GetName() const {
    return "VarFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> VarFunction::GetChildren() {
    return {arg_};
}



}