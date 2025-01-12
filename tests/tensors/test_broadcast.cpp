#include "../../statkitcpp/core/_tensor/tensor.h"
#include "../../statkitcpp/core/_tensor/shape.h"

#include <catch2/catch_test_macros.hpp>

namespace statkitcpp {

template <class T1, class T2>
void CheckVector(const std::vector<T1>& v1, const std::vector<T2>& v2) {
    CHECK(v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); i++) {
        CHECK(v1[i] == v2[i]);
    }
}

TEST_CASE("Simple broadcast") {
    {
        Tensor<float> a({1, 1});
        Tensor<float> b({1, 1});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor<float> a({2, 1});
        Tensor<float> b({1, 1});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor<float> a({2, 1});
        Tensor<double> b({1});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor<float> a({5, 4});
        Tensor<double> b({1});
        REQUIRE(a.BroadcastableTo(b));

        a = Tensor({5, 4});
        b = Tensor({4});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor<float> a({15, 3, 5});
        Tensor<double> b({15, 1, 5});
        REQUIRE(a.BroadcastableTo(b));

        a = Tensor({15, 3, 5});
        b = Tensor({3, 5});
        REQUIRE(a.BroadcastableTo(b));

        a = Tensor({15, 3, 5});
        b = Tensor({3, 1});
        REQUIRE(a.BroadcastableTo(b));
    }
    {
        Tensor a({8, 1, 6, 1});
        Tensor b({7, 1, 5});
        REQUIRE(a.BroadcastableTo(b));
    }
}

TEST_CASE("Not broadcastable") {
    {
        auto a = Tensor({3});
        auto b = Tensor({4});
        REQUIRE_FALSE(a.BroadcastableTo(b));
    }
    {
        auto a = Tensor({2, 1});
        auto b = Tensor({8, 4, 3});
        REQUIRE_FALSE(a.BroadcastableTo(b));
    }
}

TEST_CASE("Output shape after broadcast") {
    {
        auto c = BroadcastShapes({5, 4}, {1});
        std::vector<size_t> r = {5, 4};
        CheckVector(c, r);
    }
    {
        auto c = BroadcastShapes({1, 1}, {1});
        std::vector<size_t> r = {1, 1};
        CheckVector(c, r);
    }
    {
        auto c = BroadcastShapes({2, 2}, {1});
        std::vector<size_t> r = {2, 2};
        CheckVector(c, r);
    }
    {
        auto c = BroadcastShapes({2, 2}, {2});
        std::vector<size_t> r = {2, 2};
        CheckVector(c, r);
    }
    {
        auto c = BroadcastShapes({5, 5}, {5, 5});
        std::vector<size_t> r = {5, 5};
        CheckVector(c, r);
    }
    {
        auto c = BroadcastShapes({5, 4}, {4});
        std::vector<size_t> r = {5, 4};
        CheckVector(c, r);
    }
    {
        auto c = BroadcastShapes({15, 3, 5}, 
                                                        {15, 1, 5});
        std::vector<size_t> r = {15, 3, 5};
        CheckVector(c, r);
    }
    {
        auto c = BroadcastShapes({15, 3, 5}, 
                                                        {3, 5});
        std::vector<size_t> r = {15, 3, 5};
        CheckVector(c, r);
    }
    {
        auto c = BroadcastShapes({15, 3, 5}, 
                                                        {3, 1});
        std::vector<size_t> r = {15, 3, 5};
        CheckVector(c, r);
    }
}

}