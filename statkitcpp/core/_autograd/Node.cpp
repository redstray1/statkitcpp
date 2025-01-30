#include "Node.h"
#include "Tensor.h"
#include "tensor_creation_ops.h"
#include "Operations.h"

namespace statkitcpp {

void Node::AddGrad(const Tensor& to_add) {
    if (temp_grad == nullptr) {
        temp_grad = std::make_unique<Tensor>(to_add.GetShape(), to_add.GetDType());
        ops::zeros(*temp_grad);
    }
    *temp_grad = AddImpl(*temp_grad, to_add);
}

}