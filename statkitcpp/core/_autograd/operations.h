#ifndef AUTOGRAD_OPS_HEADER_H
#define AUTOGRAD_OPS_HEADER_H
#include "TensorImpl.h"
#include "autograd.h"
#include <vector>

namespace statkitcpp {


//Binary operations-------------------------------------------------------------------------------
class AddFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> lhs_;
    std::shared_ptr<TensorImpl> rhs_;
public:
    AddFunction() = default;
    ~AddFunction() {}
    Tensor Forward(const Tensor& lhs, const Tensor& rhs);
    Tensor Forward(const Tensor& lhs, const Scalar& rhs);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

class SubFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> lhs_;
    std::shared_ptr<TensorImpl> rhs_;
public:
    SubFunction() = default;
    ~SubFunction() {}
    Tensor Forward(const Tensor& lhs, const Tensor& rhs);
    Tensor Forward(const Tensor& lhs, const Scalar& rhs);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

class MulFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> lhs_;
    std::shared_ptr<TensorImpl> rhs_;
public:
    MulFunction() = default;
    ~MulFunction() {}
    Tensor Forward(const Tensor& lhs, const Tensor& rhs);
    Tensor Forward(const Tensor& lhs, const Scalar& rhs);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

class DivFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> lhs_;
    std::shared_ptr<TensorImpl> rhs_;
public:
    DivFunction() = default;
    ~DivFunction() {}
    Tensor Forward(const Tensor& lhs, const Tensor& rhs);
    Tensor Forward(const Tensor& lhs, const Scalar& rhs);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

class PowFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> lhs_;
    std::shared_ptr<TensorImpl> rhs_;
public:
    PowFunction() = default;
    ~PowFunction() {}
    Tensor Forward(const Tensor& lhs, const Tensor& rhs);
    Tensor Forward(const Tensor& lhs, const Scalar& rhs);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

//Aggregation operations-------------------------------------------------------------------------------

class SumFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> arg_;
public:
    SumFunction() = default;
    ~SumFunction() {}
    Tensor Forward(const Tensor& arg, int dim, bool keepdims);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

class ProdFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> arg_;
public:
    ProdFunction() = default;
    ~ProdFunction() {}
    Tensor Forward(const Tensor& arg, int dim, bool keepdims);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

class MeanFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> arg_;
    size_t dim_;
public:
    MeanFunction() = default;
    ~MeanFunction() {}
    Tensor Forward(const Tensor& arg, int dim, bool keepdims);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

class VarFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> arg_;
    size_t dim_;
public:
    VarFunction() = default;
    ~VarFunction() {}
    Tensor Forward(const Tensor& arg, int dim, bool keepdims);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

//Pointwise operations-------------------------------------------------------------------------------
class NegFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> arg_;
public:
    NegFunction() = default;
    ~NegFunction() {}
    Tensor Forward(const Tensor& arg);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

class ExpFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> arg_;
public:
    ExpFunction() = default;
    ~ExpFunction() {}
    Tensor Forward(const Tensor& arg);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

class LogFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> arg_;
public:
    LogFunction() = default;
    ~LogFunction() {}
    Tensor Forward(const Tensor& arg);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};


class SqrtFunction : public GradFunction {
private:
    std::shared_ptr<TensorImpl> arg_;
public:
    SqrtFunction() = default;
    ~SqrtFunction() {}
    Tensor Forward(const Tensor& arg);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

//Tensor operations-------------------------------------------------------------------------------
class ReshapeOperation : public GradFunction {
private:
    std::shared_ptr<TensorImpl> arg_;
public:
    ReshapeOperation() = default;
    ~ReshapeOperation() {}
    Tensor Forward(const Tensor& arg, const std::vector<size_t>& shape);
    std::vector<Tensor> Backward(const Tensor& grad_output) override;
    std::string GetName() const override;
    std::vector<std::shared_ptr<TensorImpl>> GetChildren() override;
};

}
#endif