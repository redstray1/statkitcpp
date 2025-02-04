#include "../../statkitcpp/core/_tensor/Tensor.h"
#include "../../statkitcpp/core/_tensor/TensorCreationOps.h"
#include "datatypes.h"
#include <iostream>

#include <catch2/catch_test_macros.hpp>

namespace statkitcpp {

TEST_CASE("Dot test") {
    {
        Tensor a = Arange(0, 5);
        Tensor b = Arange(0, 5);
        auto c = a.Dot(b);
        // std::cout << c.ToString() << std::endl;
        c = a.Dot(b);
        // std::cout << c.ToString() << std::endl;
    }
    {
        Tensor a = Full({10, 2}, 5);
        Tensor b = Full({5,3,4,2,5}, 4);
        auto c = a.MatMul(b);
    }
}

}