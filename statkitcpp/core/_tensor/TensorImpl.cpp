#include "TensorImpl.h"
#include "../errors.h"
#include "Operations.h"
#include "Scalar.h"
#include "ScalarType.h"
#include "Tensor.h"
#include "shape.h"
#include "tensor_creation_ops.h"
#include <cstddef>
#include <cassert>
#include <sys/types.h>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <iostream>
#include <stdexcept>

namespace statkitcpp {

// TensorImpl class implementation START-----------------------------------------------------
// Constructors START------------------------------------------------------------

TensorImpl::TensorImpl() {
    shape_ = {0};
    strides_ = {0};
    size_ = 0;
    storage_ = Storage();
}

TensorImpl::TensorImpl(const std::vector<size_t>& shape, ScalarType dtype, bool requires_grad) {
    shape_ = shape;
    size_ = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    strides_.resize(shape.size(), 1);
    dtype_ = dtype;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape[i + 1];
    }
    storage_ = Storage(size_ * ItemSize(dtype));
    requires_grad_ = requires_grad;
}

TensorImpl::TensorImpl(void* data, 
                  const std::vector<size_t>& shape,
                  ScalarType dtype,
                  bool requires_grad) {
    size_ = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    shape_ = shape;
    strides_.resize(shape.size(), 1);
    dtype_ = dtype;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape[i + 1];
    }
    storage_ = Storage(data, size_ * ItemSize(dtype));
    requires_grad_ = requires_grad;
}

TensorImpl::TensorImpl(const Storage& storage,
                       const std::vector<size_t>& shape,
                       ScalarType dtype, bool requires_grad) {
    size_ = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    shape_ = shape;
    strides_.resize(shape.size(), 1);
    dtype_ = dtype;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape[i + 1];
    }
    storage_ = storage;
    requires_grad_ = requires_grad;
}

TensorImpl::TensorImpl(const Scalar& scalar,
                       ScalarType dtype,
                       bool requires_grad) : storage_(ItemSize(dtype)) {
    SKPP_CHECK(scalar, dtype);
    size_ = 1;
    shape_ = {1};
    strides_ = {1};
    dtype_ = dtype;
    requires_grad_ = requires_grad;
    #define DEFINE_DTYPE(T, name) \
    case (ScalarType::name): \
        *static_cast<T*>(storage_.GetDataPtr()) = scalar.to##name(); \
        break;
    switch(dtype) {
        SCALAR_TYPES(DEFINE_DTYPE)
        default:
            throw InvalidDatatypeError{};
    }
    #undef DEFINE_DTYPE
}


// Constructors END------------------------------------------------------------

std::string TensorImpl::GetTypeName() const {
    return GetDTypeName(dtype_);
}

size_t TensorImpl::GetFlatIndex(const std::vector<size_t>& indexes) const {
    assert(indexes.size() <= shape_.size());
    size_t index = 0;
    for (size_t i = 0; i < indexes.size(); i++) {
        if (indexes[i] >= shape_[i]) {
            throw OutOfRangeError{i, indexes[i]};
        }
        index += indexes[i] * strides_[i] / GetItemSize();
    }
    return index;
}

std::vector<size_t> TensorImpl::GetIndexesFromFlat(size_t flat_index) const {
    std::vector<size_t> indexes(shape_.size(), 0);
    if (flat_index >= size_) {
        throw OutOfRangeFlatError{flat_index, size_};
    }
    for (size_t i = 0; i < shape_.size(); ++i) {
        indexes[i] = flat_index / (strides_[i] / GetItemSize());
        flat_index %= (strides_[i] / GetItemSize());
    }
    return indexes;
}

void TensorImpl::RecursiveToString(size_t depth, size_t& cur_index, std::string& result) const {
    result += '[';
    if (depth == shape_.size() - 1) {
        for (int i = 0; i < static_cast<int>(shape_[depth]); i++) {
            result += storage_.ToString(cur_index, dtype_);
            if (i < static_cast<int>(shape_[depth]) - 1) {
                result += ", ";
            }
            cur_index++;
        }
        result += ']';
        return;
    }
    std::string newline;
    for (int i = 0; i < static_cast<int>(shape_.size() - depth - 1); i++) {
        newline += '\n';
    }
    std::string shift = "        ";
    for (int i = 0; i < static_cast<int>(depth); i++) {
        shift += ' ';
    }
    for (int i = 0; i < static_cast<int>(shape_[depth]); i++) {
        RecursiveToString(depth + 1, cur_index, result);
        if (i < static_cast<int>(shape_[depth]) - 1) {
            result += "," + newline + shift;
        }
    }
    result += ']';
}

//------------------------------------------------------------------------------------
//Public methods----------------------------------------------------------------------
//------------------------------------------------------------------------------------

std::string TensorImpl::ToString() const {
    std::string shape_repr = ShapeToString(shape_);
    std::string tensor_repr;
    std::string dtype_repr = GetTypeName();
    size_t index = 0;
    RecursiveToString(0, index, tensor_repr);
    std::string result = "Tensor(" + tensor_repr + ", shape=" + shape_repr + ", dtype=" + dtype_repr;
    if (grad_fn != nullptr) {
        result += ", grad_fn=";
        result += grad_fn->GetName();
    }
    result += ")";
    return result;
}

const std::vector<size_t> TensorImpl::GetShape() const {
    return shape_;
}

void TensorImpl::SetShape(const std::vector<size_t>& shape) {
    size_t new_size =
        std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    if (new_size != size_) {
        throw ReshapeError{size_, ShapeToString(shape)};
    }
    shape_ = shape;
}

void TensorImpl::Reshape(const std::vector<size_t>& new_shape) {
    SetShape(new_shape);
}

void TensorImpl::Unsqueeze(int dim) {
    if (dim < 0) {
        dim += GetNDim();
    }
    shape_.insert(shape_.begin() + dim, 1);
    if (dim == 0) {
        strides_.insert(strides_.begin(), size_);
    } else {
        strides_.insert(strides_.begin() + dim, strides_[dim - 1]);
    }
}

const std::vector<size_t> TensorImpl::GetStrides() const {
    return strides_;
}

size_t TensorImpl::GetSize() const {
    return size_;
}

size_t TensorImpl::GetNDim() const {
    return shape_.size();
}

size_t TensorImpl::GetDim(int dim) const {
    if (dim < 0) {
        dim += GetNDim();
    }
    return shape_[dim];
}

Tensor& TensorImpl::GetGrad() {
    if (grad == nullptr) {
        throw std::runtime_error{"Something strange happened, there is no .grad in this tensor"};
    }
    return *grad;
}

void TensorImpl::AddGrad(const Tensor& to_add) {
    if (grad == nullptr) {
        grad = std::make_shared<Tensor>(GetShape(), dtype_);
        ops::zeros(*grad);
    }
    *grad = AddImpl(*grad, to_add);
}

std::shared_ptr<Node>& TensorImpl::GetAutogradNode() {
    if (autograd_node_ == nullptr) {
        autograd_node_ = std::make_shared<Node>(weak_from_this(), nullptr);
        // std::cout << "Created node " << autograd_node_ << std::endl;
    }
    return autograd_node_;
}

// TensorImpl class implementation END-----------------------------------------------------

}