#include "../../statkitcpp/core/_tensor/Tensor.h"
#include "../../statkitcpp/core/_tensor/TensorCreationOps.h"
#include "datatypes.h"

#include <catch2/catch_test_macros.hpp>

namespace statkitcpp {

TEST_CASE("Transpose") {
    {
        Tensor a = Ones({5, 5});
        auto b = a.Transpose();
    }
}

}