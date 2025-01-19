#pragma once

#include "_tensor/ScalarType.h"

namespace statkitcpp {

using Dtype = ScalarType;

constexpr auto kInt8 = ScalarType::Char;
constexpr auto kInt16 = ScalarType::Short;
constexpr auto kInt32 = ScalarType::Int;
constexpr auto kInt64 = ScalarType::Long;
constexpr auto kFloat32 = ScalarType::Float;
constexpr auto kFloat64 = ScalarType::Double;

}