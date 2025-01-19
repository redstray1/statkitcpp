#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include "ScalarType.h"

namespace statkitcpp {

class Scalar {
public:
Scalar() : Scalar(static_cast<int64_t>(0)) {}

~Scalar() {}

#define DEFINE_IMPLICIT_CTOR(type, name) \
Scalar(type vv) : Scalar(vv, true) {}

SCALAR_TYPES_NOBOOL(DEFINE_IMPLICIT_CTOR)

#undef DEFINE_IMPLICIT_CTOR

#define DEFINE_TOTYPE(type, name) \
type to##name() const {           \
    if (Tag::has_d == tag_) {     \
        return static_cast<type>(v_.d); \
    } else if (Tag::has_b == tag_) { \
        return static_cast<type>(v_.i); \
    } else if (Tag::has_i == tag_) { \
        return static_cast<type>(v_.i); \
    } \
    throw std::runtime_error{"Unknown type"}; \
}

SCALAR_TYPES(DEFINE_TOTYPE)

#undef DEFINE_TOTYPE

template <typename T>
T to() const = delete;

bool IsIntegral() const {
    return tag_ == Tag::has_i;
}   

bool IsFloatingPoint() const {
    return tag_ == Tag::has_d;
}

bool IsBoolean() const {
    return tag_ == Tag::has_b;
}

Scalar& operator=(Scalar&& other) noexcept {
    if (&other == this) {
        return *this;
    }
    MoveFrom(std::move(other));
    return *this;
}

Scalar& operator=(const Scalar& other) {
    if (&other == this) {
        return *this;
    }
    *this = Scalar(other);
    return *this;
}

template <
    typename T>
bool Equal(T num) const {
    if (IsFloatingPoint()) {
        return toDouble() == num;
    } else if (tag_ == Tag::has_i) {
        return static_cast<T>(v_.i) == num;
    } else if (IsBoolean()) {
        return false;
    } else {
        throw std::runtime_error{"Unknown scalar type."};
    }
}

bool Equal(bool num) const {
    if (IsBoolean()) {
        return static_cast<bool>(v_.i) == num;
    } else {
        return false;
    }
}

ScalarType Type() const {
    if (IsFloatingPoint()) {
        return ScalarType::Double;
    } else if (IsIntegral()) {
        return ScalarType::Long;
    } else if (IsBoolean()) {
        return ScalarType::Bool;
    } else {
        throw std::runtime_error{"Unknown scalar type."};
    }
}

Scalar(Scalar&& rhs) noexcept : tag_(rhs.tag_) {
    MoveFrom(std::move(rhs));
}

Scalar(const Scalar& rhs) : tag_(rhs.tag_), v_(rhs.v_) {
}

private:

enum class Tag { has_i, has_d, has_b };

void MoveFrom(Scalar&& rhs) noexcept {
    v_ = rhs.v_;
    tag_ = rhs.tag_;
}

Tag tag_;

union v_t {
    double d{};
    int64_t i;
    v_t() {}
} v_;

template <
    typename T,
    typename std::enable_if_t<
        std::is_integral_v<T> && !std::is_same_v<T, bool>,
        bool>* = nullptr>
    Scalar(T vv, bool) : tag_(Tag::has_i) {
        v_.i = static_cast<decltype(v_.i)>(vv);
    }

template <
    typename T,
    typename std::enable_if_t<
        !std::is_integral_v<T>,
        bool>* = nullptr>
    Scalar(T vv, bool) : tag_(Tag::has_d) {
        v_.d = static_cast<decltype(v_.d)>(vv);
    }

};

#define DEFINE_TO(T, name) \
template <> \
inline T Scalar::to<T>() const { \
    return to##name(); \
}

SCALAR_TYPES(DEFINE_TO)
#undef DEFINE_TO

void SKPP_CHECK(const Scalar& arg, ScalarType type); //NOLINT

}