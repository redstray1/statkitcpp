#include "tensor.h"
#include "config.h"

namespace statkitcpp {

// Tensor class implementation START-----------------------------------------------------
template <typename  T>
Tensor<T>::Tensor() {
    shape_ = {0};
    data_ = {};
}

template <typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shape, data_type dtype) {
    shape_ = shape;
    dtype_ = dtype;
    size_ = 1;
    for (const auto& dim : shape) {
        size_ *= dim;
    }
    data_.resize(size_);
}

template <typename T>
std::vector<uint32_t> Tensor<T>::GetShape() const {
    return shape_;
}

template <typename T>
void Tensor<T>::SetShape(const std::vector<uint32_t>& shape) {
    shape_ = shape;
}

template <typename T>
uint32_t Tensor<T>::GetSize() const {
    return size_;
}

template <typename T>
void Tensor<T>::SetRequiresGrad(bool requires_grad) {
    requires_grad_ = requires_grad;
}

template <typename T>
const bool Tensor<T>::GetRequiresGrad() const {
    return requires_grad_;
}

// Tensor class implementation END-----------------------------------------------------

template <typename T>
Tensor<T> Tensor<T>::Full(const std::vector<uint32_t>& shape, T value, data_type dtype) {
    Tensor<T> tensor(shape, dtype);
    for (int i = 0; i < tensor.size_; i++) {
        tensor.data_[i] = value;
    }
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::Ones(const std::vector<uint32_t>& shape, T value, data_type dtype) {
    return Tensor<T>::Full(shape, 1, dtype);
}

template <typename T>
Tensor<T> Tensor<T>::Zeros(const std::vector<uint32_t>& shape, T value, data_type dtype) {
    return Tensor<T>::Full(shape, 1, dtype);
}

template class Tensor<float>;
}