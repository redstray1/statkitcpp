#pragma once
#include <cstddef>
#include <vector>
#include "Tensor.h"
#include "TensorImpl.h"

namespace statkitcpp {

class TensorIterator {
private:
    using difference_type = ptrdiff_t;
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
        int Get(size_t x) const {
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
    //std::shared_ptr<TensorImpl> impl_;
    const Tensor* tensor_;
    char itemsize_;
    size_t ndim_;
    std::vector<size_t> shape_;
    char* ptr_;
    size_t ones_mask_ = 0;
    DimMask dim_mask_;
    std::vector<size_t> idxs_;
    DimMask cleared_mask_;
    
public:
    explicit TensorIterator(const Tensor& lhs);
    TensorIterator(const TensorIterator& other);
    TensorIterator& operator=(const TensorIterator& other) = default;
    bool Expired() const;
    void* Iter() const;
    void ClearLastN(const size_t& n);
    size_t GetLeading() const { return dim_mask_.GetLeading(); }
    size_t GetNDim() const { return tensor_->GetNDim(); }
    size_t GetDim(const size_t& dim) const { return tensor_->GetDim(dim); }
    void Inc(const int& dim);
    void Dec(int dim);
    // TensorIterator& operator++();
    // TensorIterator& operator+=(difference_type n);
    // TensorIterator& operator-=(difference_type n);
};

class TensorIteratorPair {
private:
    TensorIterator iter1_;
    TensorIterator iter2_;
    bool expired_ = false;
public:
    explicit TensorIteratorPair(const Tensor& lhs, const Tensor& rhs);
    TensorIteratorPair(const TensorIteratorPair& other);
    TensorIteratorPair& operator=(const TensorIteratorPair& other);
    void* IterF() const;
    void* IterS() const;
    bool Expired() const;
    TensorIteratorPair& operator++();
    TensorIteratorPair operator++(int);
    //auto operator<=>(const TensorIteratorPair&) const = default;
};

}