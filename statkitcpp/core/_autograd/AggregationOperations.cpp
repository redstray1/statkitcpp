#include "operations.h"
#include "tensor_creation_ops.h"
#include "../_tensor/operations/Operations.h"
#include "Tensor.h"
#include <iostream>
#include "shape.h"

namespace statkitcpp {


Tensor SumFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    dim_ = dim;
    keepdims_ = keepdims;
    auto output = SumImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> SumFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg(arg_->GetShape(), arg_->GetDType());
        ops::ones(darg);
        darg = MulImpl(darg, UnsqueezeImpl(grad_output, dim_));
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
    dim_ = dim;
    keepdims_ = keepdims;
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
    keepdims_ = keepdims;
    auto output = MeanImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}



std::vector<Tensor> MeanFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg(arg_->GetShape(), arg_->GetDType());
        ops::ones(darg);
        darg = MulImpl(darg, UnsqueezeImpl(grad_output, dim_));
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
    keepdims_ = keepdims;
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
        darg = MulImpl(darg, UnsqueezeImpl(grad_output, dim_));
        auto dsub = SubImpl(temp, MeanImpl(temp, dim_, true));
        darg = MulImpl(darg, AddImpl(dsub, MeanImpl(dsub, dim_, true)));
        darg = MulImpl(darg, 2.0 / static_cast<double>(arg_->GetDim(dim_) - 1));
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



Tensor MaxFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    dim_ = dim;
    keepdims_ = keepdims;
    auto output = MaxImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> MaxFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg(arg_->GetShape(), arg_->GetDType());
        ops::ones(darg);
        Tensor temp(arg_);
        darg = MulImpl(darg, UnsqueezeImpl(grad_output, dim_));
        darg = MulImpl(darg, MaxDerivImpl(temp, dim_));
        dargs[0] = darg;
    }
    return dargs;
}

std::string MaxFunction::GetName() const {
    return "MaxFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> MaxFunction::GetChildren() {
    return {arg_};
}




Tensor MinFunction::Forward(const Tensor& arg, int dim, bool keepdims) {
    bool requires_grad = arg.GetRequiresGrad();
    arg_ = arg.GetImpl();
    dim_ = dim;
    keepdims_ = keepdims;
    auto output = MinImpl(arg, dim, keepdims);
    output.SetRequiresGrad(requires_grad);
    return output;
}

std::vector<Tensor> MinFunction::Backward(const Tensor& grad_output) {
    std::vector<Tensor> dargs(1);
    if (arg_->GetRequiresGrad()) {
        Tensor darg(arg_->GetShape(), arg_->GetDType());
        ops::ones(darg);
        Tensor temp(arg_);
        darg = MulImpl(darg, UnsqueezeImpl(grad_output, dim_));
        darg = MulImpl(darg, MinDerivImpl(temp, dim_));
        dargs[0] = darg;
    }
    return dargs;
}

std::string MinFunction::GetName() const {
    return "MinFunction()";
}

std::vector<std::shared_ptr<TensorImpl>> MinFunction::GetChildren() {
    return {arg_};
}



}