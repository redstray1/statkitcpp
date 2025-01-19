#include "ScalarType.h"
#include <cstdint>
#include <stdexcept>
#include "../utils/Array.h"

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
    for (int64_t i = 0; i < static_cast<size_t>(ScalarType::NumOptions); i++) {
        inverse[i] = -1;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(kIndex2dtype.size()); i++) {
        inverse[static_cast<int64_t>(kIndex2dtype[i])] = i;
    }
    return inverse;
}

constexpr auto kDType2Index = CalcDType2Index();

ScalarType PromoteTypes(ScalarType a, ScalarType b) {
    if (a == b) {
        return a;
    }
    auto idx_a = kDType2Index[static_cast<int64_t>(a)];
    if (idx_a == -1) {
        throw std::runtime_error{"Unknown ScalarType"};
    }
    auto idx_b = kDType2Index[static_cast<int64_t>(b)];
    if (idx_b == -1) {
        throw std::runtime_error{"Unknown ScalarType"};
    }
    static constexpr std::
    array<std::array<ScalarType, kDType2Index.size()>, kDType2Index.size()>
      kPromoteTypesLookup = {{
      /*        ki1  ki2  ki4  ki8  kf4  kf8  kb1*/
      /* i1 */ {kI1, kI2, kI4, kI8, kF4, kF8, kI1},
      /* i2 */ {kI2, kI2, kI4, kI8, kF4, kF8, kI2},
      /* i4 */ {kI4, kI4, kI4, kI8, kF4, kF8, kI4},
      /* i8 */ {kI8, kI8, kI8, kI8, kF4, kF8, kI8},
      /* f4 */ {kF4, kF4, kF4, kF4, kF4, kF8, kF4},
      /* f8 */ {kF8, kF8, kF8, kF8, kF8, kF8, kF8},
      /* b1 */ {kI1, kI2, kI4, kI8, kF4, kF8, kB1},
  }};
  return kPromoteTypesLookup[idx_a][idx_b];
}

std::string GetDTypeName(ScalarType scalar_type) {
    switch (scalar_type) {
        case ScalarType::Bool:
            return "bool";
        case ScalarType::Char:
            return "int8";
        case ScalarType::Short:
            return "int16";
        case ScalarType::Int:
            return "int32";
        case ScalarType::Long:
            return "int64";
        case ScalarType::Float:
            return "float32";
        case ScalarType::Double:
            return "float64";
        default:
            return "UNKNOWN";
    }
}

}