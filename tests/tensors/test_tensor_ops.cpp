#include "../../statkitcpp/core/_tensor/Tensor.h"
#include "../../statkitcpp/core/_tensor/TensorCreationOps.h"
#include "datatypes.h"
#include <iostream>

#include <catch2/catch_test_macros.hpp>

namespace statkitcpp {

TEST_CASE("Reshape operation") {
    {
        Tensor a = Arange(0, 5);
        Tensor c = a.Reshape({5, 1});
        CHECK(c.GetDType() == kFloat32);
        std::cout << c.ToString();
    }
}

}