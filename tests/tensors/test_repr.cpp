#include "../../statkitcpp/core/_tensor/Tensor.h"
#include "../../statkitcpp/core/_tensor/TensorCreationOps.h"
#include "datatypes.h"

#include <catch2/catch_test_macros.hpp>

namespace statkitcpp {

TEST_CASE("Simple repr") {
    {
        Tensor a = Zeros({1, 1});
        auto res_repr = a.ToString();
        REQUIRE(res_repr == R"raw(Tensor([[0]], shape=(1,1), dtype=float32))raw");

        Tensor b = Zeros({1, 1}, kFloat64);
        res_repr = b.ToString();
        REQUIRE(res_repr == R"raw(Tensor([[0]], shape=(1,1), dtype=float64))raw");
    }
    {
        Tensor a = Ones({1, 1});
        auto res_repr = a.ToString();
        REQUIRE(res_repr == R"raw(Tensor([[1]], shape=(1,1), dtype=float32))raw");

        Tensor b = Ones({1, 1},kFloat64);
        res_repr = b.ToString();
        REQUIRE(res_repr == R"raw(Tensor([[1]], shape=(1,1), dtype=float64))raw");
    }
    {
        Tensor a = Ones({3});
        auto res_repr = a.ToString();
        REQUIRE(res_repr == R"raw(Tensor([1, 1, 1], shape=(3), dtype=float32))raw");
    }
    {
        Tensor a = Arange(0, 5);
        auto res_repr = a.ToString();
        REQUIRE(res_repr == R"raw(Tensor([0, 1, 2, 3, 4], shape=(5), dtype=float32))raw");
    }
    // {
    //     Tensor<float> a = Tensor<float>::Ones({2, 3});
    //     auto res_repr = a.ToString();
    //     REQUIRE(res_repr == "Tensor([[1, 1, 1],\n\t   [1, 1, 1]], shape=(2,3), dtype=float32)");
    // }
    // {
    //     Tensor<float> a = Tensor<float>::Ones({2, 2, 2});
    //     auto res_repr = a.ToString();
    //     REQUIRE(res_repr == "Tensor([[[1, 1],\n\t    [1, 1]],\n\n\t   [[1, 1],\n\t    [1, 1]]], shape=(2,2,2), dtype=float32)");
    // }
}

}