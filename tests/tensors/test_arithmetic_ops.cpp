#include "../../statkitcpp/core/_tensor/Tensor.h"
#include "../../statkitcpp/core/_tensor/TensorCreationOps.h"
#include "datatypes.h"
#include <iostream>

#include <catch2/catch_test_macros.hpp>

namespace statkitcpp {

TEST_CASE("Add operation") {
    {
        Tensor a = Ones({2, 3});
        Tensor b = Ones({1, 3});
        Tensor c = a.Add(b);
        CHECK(c.GetDType() == kFloat32);
    }
    {
        Tensor a = Arange(0, 5);
        Tensor b = Full({1, 5}, 0.33);
        Tensor c = a.Add(b);
        CHECK(c.GetDType() == kFloat32);
        std::cout << c.ToString();
    }
}

}