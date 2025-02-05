#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>
#include "Operations.h"
#include "Tensor.h"
#include "TensorIndex.h"
#include "errors.h"
#include "indexing_vectorized.h"
#include "shape.h"

namespace statkitcpp {

Tensor IndexingImpl(const Tensor& arg, const std::vector<TensorIndex>& indices) {
    
    Tensor output(GetShapeFromIndexing(indices, arg.GetShape()), arg.GetDType());
    char* ptr = arg.data_ptr<char>();
    size_t last_idx = 0;
    for (; last_idx < indices.size(); last_idx++) {
        if (!indices[last_idx].IsIndex()) {
            break;
        } else {
            ptr += indices[last_idx].GetIndex() * arg.GetStride(last_idx) * arg.GetItemSize();
        }
    }
    if (last_idx == indices.size()) {
        memcpy(output.GetDataPointer(), ptr, output.GetItemSize() * output.GetSize());
        return output;
    }
    if (last_idx == indices.size() - 1 && (indices[last_idx].GetSlice().step == 1)) {
        ptr += indices.back().GetSlice().start * arg.GetStride( last_idx) * arg.GetItemSize();
        memcpy(output.GetDataPointer(), ptr, output.GetSize() * output.GetItemSize());
        return output;
    }
    throw NotImplemetedError{"Indexing not implemeted for this case yet."};
}

Tensor& IndexingPutImpl(Tensor& arg, const std::vector<TensorIndex>& indices, const Tensor& other) {
    if (arg.GetRequiresGrad()) {
        std::logic_error{"You want to assign tensor to indices of tensor with requires_grad=True. Are you sure?"};
    }
    auto shape = GetShapeFromIndexing(indices, arg.GetShape());
    size_t size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    if (size != other.GetSize()) {
        throw IndexAssignError{other.GetSize(), size};
    }
    if (other.GetDType() == arg.GetDType()) {
        char* ptr = arg.data_ptr<char>();
        size_t last_idx = 0;
        for (; last_idx < indices.size(); last_idx++) {
            if (!indices[last_idx].IsIndex()) {
                break;
            } else {
                ptr += indices[last_idx].GetIndex() * arg.GetStride(last_idx) * arg.GetItemSize();
            }
        }
        if (last_idx == indices.size()) {
            memcpy(ptr, other.GetDataPointer(), size * arg.GetItemSize());
        }
        if (last_idx == indices.size() - 1 && (indices[last_idx].GetSlice().step == 1)) {
            ptr += indices.back().GetSlice().start * arg.GetStride(indices.size() - 1) * arg.GetItemSize();
            memcpy(ptr, other.GetDataPointer(), size * arg.GetItemSize());
        }
        return arg;
    } else {
        throw NotImplemetedError{"Indexing assign for different types not implemeted yet."};
    }
}

Tensor& IndexingPutImpl(Tensor& arg, const std::vector<TensorIndex>& indices, const Scalar& other) {
    if (arg.GetRequiresGrad()) {
        std::logic_error{"You want to assign tensor to indices of tensor with requires_grad=True. Are you sure?"};
    }
    auto new_shape = GetShapeFromIndexing(indices, arg.GetShape());
    auto size = std::reduce(new_shape.begin(), new_shape.end(), 1, std::multiplies());
    if (other.Type() == arg.GetDType()) {
        char* ptr = arg.data_ptr<char>();
        size_t last_idx = 0;
        for (; last_idx < indices.size(); last_idx++) {
            if (!indices[last_idx].IsIndex()) {
                break;
            } else {
                ptr += indices[last_idx].GetIndex() * arg.GetStride(last_idx) * arg.GetItemSize();
            }
        }
        if (last_idx == indices.size()) {
            vec::assign_scalar(ptr, arg.GetDType(), other.ptr(), other.Type(), size);
        }
        if (last_idx == indices.size() - 1 && (indices[last_idx].GetSlice().step == 1)) {
            ptr += indices.back().GetSlice().start * arg.GetStride(indices.size() - 1) * arg.GetItemSize();
            vec::assign_scalar(ptr, arg.GetDType(), other.ptr(), other.Type(), size);
        }
        return arg;
    } else {
        throw NotImplemetedError{"Indexing assign for different types not implemeted yet."};
    }
}

}