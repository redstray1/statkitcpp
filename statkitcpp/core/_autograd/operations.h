#ifndef AUTOGRAD_OPS_HEADER_H
#define AUTOGRAD_OPS_HEADER_H
#include <memory>
#include "autograd.h"

namespace statkitcpp {

class AddFunction : public GradFunction {
private:
    std::shared_ptr<Tensor> lhs_;
    std::shared_ptr<Tensor> rhs_;
public:
    AddFunction() = default;
    ~AddFunction() {}
    Tensor Forward(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha);
    Tensor Backward(const Tensor& grad_output, const Tensor& output) override;
};

class SubFunction : public GradFunction {
private:
    std::shared_ptr<Tensor> lhs_;
    std::shared_ptr<Tensor> rhs_;
public:
    SubFunction() = default;
    ~SubFunction() {}
    Tensor Forward(const Tensor& lhs, const Tensor& rhs, const Scalar& alpha);
    Tensor Backward(const Tensor& grad_output, const Tensor& output) override;
};

class MulFunction : public GradFunction {
private:
    std::shared_ptr<Tensor> lhs_;
    std::shared_ptr<Tensor> rhs_;
public:
    MulFunction() = default;
    ~MulFunction() {}
    Tensor Forward(const Tensor& lhs, const Tensor& rhs);
    Tensor Backward(const Tensor& grad_output, const Tensor& output) override;
};

class DivFunction : public GradFunction {
private:
    std::shared_ptr<Tensor> lhs_;
    std::shared_ptr<Tensor> rhs_;
public:
    DivFunction() = default;
    ~DivFunction() {}
    Tensor Forward(const Tensor& lhs, const Tensor& rhs);
    Tensor Backward(const Tensor& grad_output, const Tensor& output) override;
};

class SumFunction : public GradFunction {
private:
    std::shared_ptr<Tensor> arg_;
public:
    SumFunction() = default;
    ~SumFunction() {}
    Tensor Forward(const Tensor& arg, int dim, bool keepdims);
    Tensor Backward(const Tensor& grad_output, const Tensor& output) override;
};

class MeanFunction : public GradFunction {
private:
    std::shared_ptr<Tensor> arg_;
public:
    MeanFunction() = default;
    ~MeanFunction() {}
    Tensor Forward(const Tensor& arg, int dim, bool keepdims);
    Tensor Backward(const Tensor& grad_output, const Tensor& output) override;
};

class VarFunction : public GradFunction {
private:
    std::shared_ptr<Tensor> arg_;
public:
    VarFunction() = default;
    ~VarFunction() {}
    Tensor Forward(const Tensor& arg, int dim, bool keepdims);
    Tensor Backward(const Tensor& grad_output, const Tensor& output) override;
};

}
#endif