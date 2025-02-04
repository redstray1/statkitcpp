#pragma once

#include <cstddef>

struct DimMask {
    size_t mask = 0;
    size_t ndim;
    DimMask() : mask(0), ndim(0) {}
    DimMask(size_t ndim) : mask(0), ndim(ndim) {}
    void Set(size_t x) {
        mask |= 1 << (ndim - x - 1);
    }
    void Unset(size_t x) {
        mask -= (1 << (ndim - x - 1));
    }
    bool Get(size_t x) const {
        return (mask >> (ndim - x - 1)) & 1;
    }
    void ClearLastN(size_t x) {
        mask -= (1 << x) - 1;
    }
    void SetLastN(size_t x) {
        mask |= (1 << x) - 1;
    }
    size_t GetLeading() const {
        return __builtin_ctz(mask + 1);
    }
    void operator|=(size_t x) {
        mask |= x;
    }
    void operator=(const size_t& mask1) {
        mask = mask1;
    }
};