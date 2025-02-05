#ifndef SLICE_HEADER_H
#define SLICE_HEADER_H
#include <cstddef>
#include "../errors.h"

namespace statkitcpp {

struct Slice {
    size_t start;
    size_t end;
    size_t step;

    constexpr Slice() = default;

    constexpr Slice(size_t end) : start(0), end(end), step(1) {
    }

    constexpr Slice(size_t start, size_t end, size_t step = 1)
        : start(start), end(end), step(step) {
        if (step == 0) {
            throw SliceError{"Step size cannot be zero."};
        }
    }
    size_t Length() const { 
        if (start >= end) {
            return 0;
        }
        return (end - start) / step;
    }
    Slice(const Slice& other) = default;
    Slice(Slice&& other) = default;
    Slice& operator=(const Slice& other) = default;
    Slice& operator=(Slice&& other) = default;
 };

}
 #endif