#pragma once

#include "ScalarType.h"
#include "Array.h"

namespace statkitcpp {


constexpr auto kI1 = ScalarType::Char;
constexpr auto kI2 = ScalarType::Short;
constexpr auto kI4 = ScalarType::Int;
constexpr auto kI8 = ScalarType::Long;
constexpr auto kF4 = ScalarType::Float;
constexpr auto kF8 = ScalarType::Double;
constexpr auto kB1 = ScalarType::Bool;

constexpr auto kIndex2dtype = ArrayOf<
    ScalarType>(kI1, kI2, kI4, kI8, kF4, kF8, kB1);

constexpr std::array<int64_t, static_cast<size_t>(ScalarType::NumOptions)>
CalcDType2Index() {
    std::array<int64_t, static_cast<size_t>(ScalarType::NumOptions)> inverse = {};
    for (int64_t i = 0; i < static_cast<int64_t>(ScalarType::NumOptions); i++) {
        inverse[i] = -1;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(kIndex2dtype.size()); i++) {
        inverse[static_cast<int64_t>(kIndex2dtype[i])] = i;
    }
    return inverse;
}

constexpr auto kDType2Index = CalcDType2Index();

}