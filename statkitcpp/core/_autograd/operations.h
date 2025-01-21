#ifndef AUTOGRAD_OPS_HEADER_H
#define AUTOGRAD_OPS_HEADER_H
#include "autograd.h"
#include <vector>

namespace statkitcpp {


//Binary operations-------------------------------------------------------------------------------
class AddFunction : public GradFunction {
private:
    Tensor* lhs_;
    Tensor* rhs_;
public:
    AddFunction() = default;
    ~AddFunction() {}
    Tensor Forward(Tensor& lhs, Tensor& rhs);
    Tensor Forward(Tensor& lhs, const Scalar& rhs);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

class SubFunction : public GradFunction {
private:
    Tensor* lhs_;
    Tensor* rhs_;
public:
    SubFunction() = default;
    ~SubFunction() {}
    Tensor Forward(Tensor& lhs, Tensor& rhs);
    Tensor Forward(Tensor& lhs, const Scalar& rhs);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

class MulFunction : public GradFunction {
private:
    Tensor* lhs_;
    Tensor* rhs_;
public:
    MulFunction() = default;
    ~MulFunction() {}
    Tensor Forward(Tensor& lhs, Tensor& rhs);
    Tensor Forward(Tensor& lhs, const Scalar& rhs);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

class DivFunction : public GradFunction {
private:
    Tensor* lhs_;
    Tensor* rhs_;
public:
    DivFunction() = default;
    ~DivFunction() {}
    Tensor Forward(Tensor& lhs, Tensor& rhs);
    Tensor Forward(Tensor& lhs, const Scalar& rhs);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

class PowFunction : public GradFunction {
private:
    Tensor* lhs_;
    Tensor* rhs_;
public:
    PowFunction() = default;
    ~PowFunction() {}
    Tensor Forward(Tensor& lhs, Tensor& rhs);
    Tensor Forward(Tensor& lhs, const Scalar& rhs);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

//Aggregation operations-------------------------------------------------------------------------------

class SumFunction : public GradFunction {
private:
    Tensor* arg_;
public:
    SumFunction() = default;
    ~SumFunction() {}
    Tensor Forward(Tensor& arg, int dim, bool keepdims);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

class ProdFunction : public GradFunction {
private:
    Tensor* arg_;
public:
    ProdFunction() = default;
    ~ProdFunction() {}
    Tensor Forward(Tensor& arg, int dim, bool keepdims);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

class MeanFunction : public GradFunction {
private:
    Tensor* arg_;
public:
    MeanFunction() = default;
    ~MeanFunction() {}
    Tensor Forward(Tensor& arg, int dim, bool keepdims);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

class VarFunction : public GradFunction {
private:
    Tensor* arg_;
public:
    VarFunction() = default;
    ~VarFunction() {}
    Tensor Forward(Tensor& arg, int dim, bool keepdims);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

//Pointwise operations-------------------------------------------------------------------------------
class NegFunction : public GradFunction {
private:
    Tensor* arg_;
public:
    NegFunction() = default;
    ~NegFunction() {}
    Tensor Forward(Tensor& arg);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

class ExpFunction : public GradFunction {
private:
    Tensor* arg_;
public:
    ExpFunction() = default;
    ~ExpFunction() {}
    Tensor Forward(Tensor& arg);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

class LogFunction : public GradFunction {
private:
    Tensor* arg_;
public:
    LogFunction() = default;
    ~LogFunction() {}
    Tensor Forward(Tensor& arg);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};


class SqrtFunction : public GradFunction {
private:
    Tensor* arg_;
public:
    SqrtFunction() = default;
    ~SqrtFunction() {}
    Tensor Forward(Tensor& arg);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

//Tensor operations-------------------------------------------------------------------------------
class ReshapeOperation : public GradFunction {
private:
    Tensor* arg_;
public:
    ReshapeOperation() = default;
    ~ReshapeOperation() {}
    Tensor Forward(Tensor& arg, const std::vector<size_t>& shape);
    void Backward(const Tensor& grad_output, const Tensor& output) override;
    std::string GetName() const override;
};

}
#endif