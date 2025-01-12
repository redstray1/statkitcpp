#ifndef SLICE_HEADER_H
#define SLICE_HEADER_H
#include <cstdint>
#include "../errors.h"

struct Slice {
    uint32_t start;
    uint32_t end;
    uint32_t step;

    constexpr Slice() = default;

    constexpr Slice(uint32_t end) : start(0), end(end), step(1) {
    }

    constexpr Slice(uint32_t start, uint32_t end, uint32_t step = 1)
        : start(start), end(end), step(step) {
        if (step == 0) {
            throw SliceError{"step size cannot be zero."};
        }
    }
 };
 #endif

