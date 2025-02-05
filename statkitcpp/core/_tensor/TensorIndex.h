#pragma once

#include "Slice.h"

namespace statkitcpp {

class TensorIndex {
private:
    bool has_index_ = false;
    bool has_slice_ = false;
    union ti_t {
        size_t idx{};
        Slice slice;
        ti_t() {}
    } tensor_index_;
public:
    TensorIndex() {}
    TensorIndex(const TensorIndex& other) = default;
    TensorIndex(TensorIndex&& other) = default;
    TensorIndex& operator=(const TensorIndex& other) = default;
    TensorIndex& operator=(TensorIndex&& other) = default;
    TensorIndex(size_t idx) {
        has_index_ = true;
        has_slice_ = false;
        tensor_index_.idx = idx;
    }

    TensorIndex(const Slice& slice) {
        has_slice_ = true;
        has_index_ = false;
        tensor_index_.slice = slice;
    }
    bool IsIndex() const { return has_index_; }
    bool IsSlice() const { return has_slice_; }

    Slice GetSlice() const {
        return tensor_index_.slice;
    }
    size_t GetIndex() const {
        return tensor_index_.idx;
    }
};

}