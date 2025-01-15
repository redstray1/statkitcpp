#include "tensor.h"
#include "../errors.h"
#include "../utils/utils.h"
#include "shape.h"
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <sys/types.h>
#include <functional>
#include <initializer_list>
#include <numeric>

namespace statkitcpp {

// Tensor class implementation START-----------------------------------------------------
// Constructors START------------------------------------------------------------
template <typename  T>
Tensor<T>::Tensor() {
    shape_ = {0};
    strides_ = {0};
    size_ = 0;
    data_ = {};
}

template <typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shape, bool requires_grad) {
    shape_ = shape;
    requires_grad_ = requires_grad;
    size_ = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    strides_.resize(shape.size(), sizeof(T));
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape[i + 1];
    }
    data_.resize(size_);
}

template <typename T>
Tensor<T>::Tensor(T* data, 
                  const std::vector<uint32_t>& shape,
                  bool requires_grad) {
    size_ = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    shape_ = shape;
    strides_.resize(shape.size(), sizeof(T));
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape[i + 1];
    }
    requires_grad_ = requires_grad;
    data_.assign(data, data + size_);
}

template <typename T>
Tensor<T>::Tensor(std::initializer_list<T> data,
                  const std::vector<uint32_t>& shape,
                  bool requires_grad) : data_(data) {
    size_ = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    if (size_ != data_.size()) {
        throw ReshapeError{std::to_string(data_.size()), ShapeToString(shape)};
    }
    shape_ = shape;
    strides_.resize(shape.size(), sizeof(T));
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape[i + 1];
    }
    requires_grad_ = requires_grad;
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& other) {
    shape_ = other.shape_;
    data_ = other.data_;
    size_ = other.size_;
    strides_ = other.strides_;
    requires_grad_ = other.requires_grad_;
}

template <typename T>
Tensor<T>::Tensor(Tensor<T>&& other) {
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
    size_ = std::move(other.size_);
    strides_ = std::move(other.strides_);
    requires_grad_ = std::move(other.requires_grad_);
}

// Constructors END------------------------------------------------------------

template <typename T>
std::string Tensor<T>::GetTypeName() const {
    std::string typeid_name = typeid(T).name();
    if (typeid_name == "f") {
        return "float32";
    } else if (typeid_name == "d") {
        return "float64";
    } else {
        throw InvalidDatatypeError{};
    }
}

template <typename T>
uint32_t Tensor<T>::GetFlatIndex(const std::vector<uint32_t>& indexes) const {
    assert(indexes.size() <= shape_.size());
    uint32_t size_multiplier = 1;
    for (uint32_t i = shape_.size() - 1; i >= indexes.size(); i--) {
        size_multiplier *= shape_[i];
    }
    uint32_t index = 0;
    for (uint32_t i = indexes.size() - 1; i >= 0; i--) {
        if (indexes[i] >= shape_[i]) {
            throw OutOfRangeError{i, indexes[i]};
        }
        index += indexes[i] * size_multiplier;
        size_multiplier *= shape_[i];
    }
    return index;
}

template <typename T>
std::vector<uint32_t> Tensor<T>::GetIndexesFromFlat(uint32_t flat_index) const {
    std::vector<uint32_t> indexes(shape_.size(), 0);
    uint32_t size_multiplier = size_;
    if (flat_index >= size_multiplier) {
        throw OutOfRangeFlatError{flat_index};
    }
    for (size_t i = 0; i < shape_.size(); ++i) {
        size_multiplier /= shape_[i];
        indexes[i] = flat_index / size_multiplier;
        flat_index %= size_multiplier;
    }
    return indexes;
}

template <typename T>
template <class BinaryOperation>
Tensor<T> Tensor<T>::ApplyBroadcastOp(const Tensor<T>& lhs, const Tensor<T>& rhs,
                                      BinaryOperation op) {
    if (!IsBroadcastable(lhs.shape_, rhs.shape_)) {
        throw BroadcastError{lhs.ShapeToString(), rhs.ShapeToString()};
    }                                   
    std::vector<uint32_t> output_shape;
    for (int i = std::max(lhs.shape_.size(), rhs.shape_.size()) - 1; i >= 0; i--) {
        uint32_t lhs_dim = i < static_cast<int>(lhs.shape_.size()) ? lhs.shape_[i] : 1;
        uint32_t rhs_dim = i < static_cast<int>(rhs.shape_.size()) ? rhs.shape_[i] : 1;
        output_shape.push_back(std::max(lhs_dim, rhs_dim));
    }
    std::reverse(output_shape.begin(), output_shape.end());
    
    Tensor<T> output(output_shape);

    for (size_t i = 0; i < output.size_; ++i) {
        std::vector<uint32_t> output_indexes = output.GetIndexesFromFlat(i);
        std::vector<uint32_t> lhs_indexes(output_indexes);
        std::vector<uint32_t> rhs_indexes(output_indexes);
        for (int j = static_cast<int>(lhs.shape_.size()) - 1; j >= 0; --j) {
            if (j >= output_shape.size() || lhs.shape_[j] == 1) {
                lhs_indexes[j] = 0;
            }
        }
        for (int j = static_cast<int>(rhs.shape_.size()) - 1; j >= 0; --j) {
            if (j >= output_shape.size() || rhs.shape_[j] == 1) {
                rhs_indexes[j] = 0;
            }
        }
        uint32_t lhs_flat_index = lhs.GetFlatIndex(lhs_indexes);
        uint32_t rhs_flat_index = rhs.GetFlatIndex(rhs_indexes);

        output.data_[i] = op(lhs.data_[lhs_flat_index], rhs.data_[rhs_flat_index]);
    } 
    return output;
}

template <typename T>
void Tensor<T>::RecursiveToString(uint32_t depth, uint32_t& cur_index, std::string& result) const {
    result += '[';
    if (depth == shape_.size() - 1) {
        for (int i = 0; i < shape_[depth]; i++) {
            result += FloatRepr(data_[cur_index]);
            if (i < shape_[depth] - 1) {
                result += ", ";
            }
            cur_index++;
        }
        result += ']';
        return;
    }
    std::string newline;
    for (int i = 0; i < shape_.size() - depth - 1; i++) {
        newline += '\n';
    }
    std::string shift = "        ";
    for (int i = 0; i < depth; i++) {
        shift += ' ';
    }
    for (int i = 0; i < shape_[depth]; i++) {
        RecursiveToString(depth + 1, cur_index, result);
        if (i < shape_[depth] - 1) {
            result += "," + newline + shift;
        }
    }
    result += ']';
}

template <typename T>
bool Tensor<T>::BroadcastableTo(const Tensor& other) {
    return IsBroadcastable(shape_, other.shape_);
}

template <typename T>
std::string Tensor<T>::ToString() const {
    std::string shape_repr = ShapeToString(shape_);
    std::string tensor_repr;
    std::string dtype_repr = GetTypeName();
    uint32_t index = 0;
    RecursiveToString(0, index, tensor_repr);
    std::string result = "Tensor(" + tensor_repr + ", shape=" + shape_repr + ", dtype=" + dtype_repr + ")";
    return result;
}

template <typename T>
std::vector<uint32_t> Tensor<T>::GetShape() const {
    return shape_;
}

template <typename T>
void Tensor<T>::SetShape(const std::vector<uint32_t>& shape) {
    auto new_size =
        std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    if (new_size != size_) {
        throw ReshapeError{std::to_string(size_), ShapeToString(shape)};
    }
    shape_ = shape;
}

template <typename T>
void Tensor<T>::Reshape(const std::vector<uint32_t>& new_shape) {
    SetShape(new_shape);
}

template <typename T>
std::vector<uint32_t> Tensor<T>::GetStrides() const {
    return strides_;
}

template <typename T>
uint32_t Tensor<T>::GetSize() const {
    return size_;
}

template <typename T>
uint32_t Tensor<T>::GetNDim() const {
    return shape_.size();
}

template <typename T>
void Tensor<T>::SetRequiresGrad(bool requires_grad) {
    requires_grad_ = requires_grad;
}

template <typename T>
bool Tensor<T>::GetRequiresGrad() const {
    return requires_grad_;
}

// template <typename T>
// Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& rhs) {
//     *this = *this + rhs;
//     return *this;
// }

// template <typename T>
// Tensor<T> Tensor<T>::operator+(const Tensor<T>& rhs) {
//     return Tensor<T>::ApplyBroadcastOp(*this, rhs, [&](T a, T b) { return a + b; });
// }

// Tensor class implementation END-----------------------------------------------------

template <typename T>
Tensor<T> Tensor<T>::Full(const std::vector<uint32_t>& shape, const T& value) {
    Tensor<T> tensor;
    tensor.shape_ = shape;
    tensor.strides_.resize(shape.size(), sizeof(T));
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        tensor.strides_[i] = tensor.strides_[i + 1] * shape[i + 1];
    }
    tensor.size_ = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
    tensor.data_.resize(tensor.size_, value);
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::Ones(const std::vector<uint32_t>& shape) {
    return Tensor<T>::Full(shape, 1);
}

template <typename T>
Tensor<T> Tensor<T>::Zeros(const std::vector<uint32_t>& shape) {
    return Tensor<T>::Full(shape, 0);
}

template class Tensor<float>;
template class Tensor<double>;

}