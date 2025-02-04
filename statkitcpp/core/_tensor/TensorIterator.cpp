#include "TensorIterator.h"
#include "Tensor.h"
#include "TensorImpl.h"
#include "utils.h"
#include <cassert>
#include <iostream>

namespace statkitcpp {

TensorIterator::TensorIterator(const Tensor& lhs) {
    tensor_ = &lhs;
    itemsize_ = lhs.GetItemSize();
    ndim_ = lhs.GetNDim();
    dim_mask_ = DimMask(ndim_);
    cleared_mask_ = DimMask(ndim_);
    ptr_ = static_cast<char*>(lhs.GetDataPointer());
    idxs_.resize(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
        if (tensor_->GetDim(i) == 1) {
            ones_mask_ += (1 << (ndim_ - i - 1));
        }
    }
    dim_mask_ |= ones_mask_;
}

TensorIterator::TensorIterator(const TensorIterator& other) : tensor_(other.tensor_) {
    dim_mask_ = other.dim_mask_;
    ptr_ = other.ptr_;
    idxs_ = other.idxs_;
}

// TensorIterator& TensorIterator::operator=(const TensorIterator& other) {
//     tensor_ = other.tensor_;
//     dim_mask_ = other.dim_mask_;
//     ndim_ 
//     ptr_ = other.ptr_;
//     idxs_ = other.idxs_;
//     return *this;
// }

void* TensorIterator::Iter() const {
    return ptr_;
}

void TensorIterator::ClearLastN(const size_t& n) {
    if (n == 0) { return; }
    assert(dim_mask_.GetLeading() >= n);
    dim_mask_.ClearLastN(n);
    dim_mask_ |= ones_mask_;
    cleared_mask_.SetLastN(n);
    ptr_ -= itemsize_ * (tensor_->GetStride(ndim_ - n) * tensor_->GetDim(ndim_ - n) - 1);
}

void TensorIterator::Inc(const int& dim) {
    if (cleared_mask_.Get(dim)) {
        idxs_[dim] = 0;
        cleared_mask_.Unset(dim);
    }
    idxs_[dim]++;
    if (idxs_[dim] == tensor_->GetDim(dim) - 1) {
        dim_mask_.Set(dim);
    }
    ptr_ += itemsize_ * tensor_->GetStride(dim);
}

void TensorIterator::Dec(int dim) {
    idxs_[dim]--;
    ptr_ -= itemsize_ * tensor_->GetStride(dim);
}



TensorIteratorPair::TensorIteratorPair(const Tensor& lhs, const Tensor& rhs)
    : iter1_(lhs), iter2_(rhs) {
}

TensorIteratorPair::TensorIteratorPair(const TensorIteratorPair& other)
    : iter1_(other.iter1_), iter2_(other.iter2_) {
}

TensorIteratorPair& TensorIteratorPair::operator=(const TensorIteratorPair& other) {
    if (this == &other) {
        return *this;
    }
    iter1_ = other.iter1_;
    iter2_ = other.iter2_;
    return *this;
}

void* TensorIteratorPair::IterF() const {
    return iter1_.Iter();
}

void* TensorIteratorPair::IterS() const {
    return iter2_.Iter();
}

bool TensorIteratorPair::Expired() const {
    return expired_;
}

TensorIteratorPair& TensorIteratorPair::operator++() {
    auto leading1 = iter1_.GetLeading();
    auto leading2 = iter2_.GetLeading();
    auto common_leading = std::min(leading1, leading2);

    // std::cout << leading1 << " :: " << leading2 << '\n';
    if (common_leading < iter1_.GetNDim() && common_leading < iter2_.GetNDim()) {
        leading1 = common_leading;
        leading2 = common_leading;
    }
    iter1_.ClearLastN(leading1);
    iter2_.ClearLastN(leading2);
    int pt1 = static_cast<int>(iter1_.GetNDim()) - leading1 - 1;
    int pt2 = static_cast<int>(iter2_.GetNDim()) - leading2 - 1;
    // std::cout << pt1 << ' ' << pt2 << std::endl;
    // PrintVector(iter1_.idxs_);
    // PrintVector(iter2_.idxs_);
    // std::cout << '\n';
    if (pt1 < 0 && pt2 < 0) {
        expired_ = true;
        return *this;
    }
    iter1_.Inc(pt1);
    iter2_.Inc(pt2);
    // if (pt1 >= 0 && iter1_.GetDim(pt1) > 1) {
    //     iter1_.Inc(pt1);
    // }
    // if (pt2 >= 0 && iter2_.GetDim(pt2) > 1) {
    //     iter2_.Inc(pt2);
    // }
    return *this;
}

TensorIteratorPair TensorIteratorPair::operator++(int) {
    auto tmp = *this;
    ++(tmp);
    return tmp;
}

}