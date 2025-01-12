#include <vector>
#include <memory>
#include "../_tensor/tensor.h"
#include "../_tensor/variable.h"
#include "autograd.h"

namespace statkitcpp {

class AddFunction : public GradFunction {
private:
    std::shared_ptr<Variable> lhs_;
    std::shared_ptr<Variable> rhs_;
public:
    template <class T1, class T2>
    auto Forward(const Tensor<T1>& lhs, const Tensor<T2>& rhs);
    template <class T>
    auto Backward(const Tensor<T>& grad_output);
};

}