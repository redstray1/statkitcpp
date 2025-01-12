#include "operations.h"
#include <memory>

namespace statkitcpp {

template <class T1, class T2>
auto AddFunction::Forward(const Tensor<T1>& lhs, const Tensor<T2>& rhs) {
    lhs_ = std::make_shared(lhs);
    rhs_ = std::make_shared(rhs);
}

}