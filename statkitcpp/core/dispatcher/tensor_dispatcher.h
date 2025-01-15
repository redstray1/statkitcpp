#ifndef TENSOR_BINDING_HEADER_H
#define TENSOR_BINDING_HEADER_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include "../_tensor/shape.h"
#include "../_tensor/tensor.h"
#include "../errors.h"

namespace py = pybind11;

namespace statkitcpp {

class TensorDispatcher {
private:
    std::shared_ptr<Variable> tensor_;
    std::string dtype_;
public:
    TensorDispatcher();
    TensorDispatcher(const std::vector<uint32_t>& shape, py::str dtype, bool requires_grad = true);
    TensorDispatcher(const TensorDispatcher& other) = default;
    TensorDispatcher(TensorDispatcher&& other) = default;
    TensorDispatcher(std::shared_ptr<Variable> tensor, const std::string& dtype = "float32")
        : tensor_(std::move(tensor)), dtype_(dtype) {}
    TensorDispatcher(py::buffer b);
    ~TensorDispatcher() {}

    TensorDispatcher& operator=(const TensorDispatcher& other) = default;
    TensorDispatcher& operator=(TensorDispatcher&& other) = default;

    std::string ToString() const;
    bool BroadcastableTo(const TensorDispatcher& other);

    std::vector<uint32_t> GetShape() const;
    void SetShape(const std::vector<uint32_t>& shape);
    void Reshape(const std::vector<uint32_t>& new_shape);

    std::vector<uint32_t> GetStrides() const;

    uint32_t GetSize() const;
    
    uint32_t GetNDim() const;

    void SetRequiresGrad(bool requires_grad);
    bool GetRequiresGrad() const;


    auto GetDataType() const;
    void* GetDataPointer();
    std::string GetDType() const;
    uint32_t GetItemSize() const;
    uint32_t GetNBytes() const;

    TensorDispatcher Sum(int dim, bool keepdims) const;
    TensorDispatcher Mean(int dim, bool keepdims) const;
    TensorDispatcher Var(int dim, bool keepdims) const;
};  

TensorDispatcher::TensorDispatcher() {
    dtype_ = "float32";
    tensor_ = std::make_shared<Tensor<float>>();
}

TensorDispatcher::TensorDispatcher(const std::vector<uint32_t>& shape, py::str dtype, bool requires_grad) {
    std::string dtype_str = static_cast<std::string>(dtype);
    if (dtype_str == "float32") {
        tensor_ = std::make_shared<Tensor<float>>(shape, requires_grad);
        dtype_ = dtype_str;
    } else if (dtype_str == "float64") {
        tensor_ = std::make_shared<Tensor<double>>(shape, requires_grad);
        dtype_ = dtype_str;
    } else {
        throw InvalidDatatypeError{};
    }
}

TensorDispatcher::TensorDispatcher(py::buffer b) {
    py::buffer_info info = b.request();
    std::vector<uint32_t> shape;
    std::transform(info.shape.begin(), info.shape.end(), std::back_inserter(shape), [](const int value)
    {
        return static_cast<uint32_t>(value);
    });
    if (info.item_type_is_equivalent_to<float>()) {
        tensor_ = std::make_shared<Tensor<float>>(static_cast<float*>(info.ptr), shape);
        dtype_ = "float32";
    } else if (info.item_type_is_equivalent_to<double>())  {
        tensor_ = std::make_shared<Tensor<double>>(static_cast<double*>(info.ptr), shape);
        dtype_ = "float64";
    } else {
        throw InvalidDatatypeError{};
    }
}

std::string TensorDispatcher::ToString() const {
    return tensor_->ToString();
}

bool TensorDispatcher::BroadcastableTo(const TensorDispatcher& other) {
    return IsBroadcastable(tensor_->GetShape(), other.GetShape());
}

std::vector<uint32_t> TensorDispatcher::GetShape() const {
    return tensor_->GetShape();
}

void TensorDispatcher::SetShape(const std::vector<uint32_t>& new_shape) {
    tensor_->SetShape(new_shape);
}

void TensorDispatcher::Reshape(const std::vector<uint32_t>& new_shape) {
    tensor_->Reshape(new_shape);
} 

std::vector<uint32_t> TensorDispatcher::GetStrides() const {
    return tensor_->GetStrides();
}

uint32_t TensorDispatcher::GetSize() const {
    return tensor_->GetSize();
}

uint32_t TensorDispatcher::GetNDim() const {
    return tensor_->GetNDim();
}

void TensorDispatcher::SetRequiresGrad(bool requires_grad) {
    tensor_->SetRequiresGrad(requires_grad);
}

bool TensorDispatcher::GetRequiresGrad() const {
    return tensor_->GetRequiresGrad();
}

void* TensorDispatcher::GetDataPointer() {
    return tensor_->GetDataPointer();
}

std::string TensorDispatcher::GetDType() const {
    return dtype_;
}

uint32_t TensorDispatcher::GetItemSize() const {
    return tensor_->GetItemSize();
}

uint32_t TensorDispatcher::GetNBytes() const {
    return tensor_->GetNBytes();
}

TensorDispatcher TensorDispatcher::Sum(int dim, bool keepdims) const {
    auto var_ptr = tensor_->ISum(dim, keepdims);
    return TensorDispatcher(var_ptr, dtype_);
}

TensorDispatcher TensorDispatcher::Mean(int dim, bool keepdims) const {
    auto var_ptr = tensor_->IMean(dim, keepdims);
    return TensorDispatcher(var_ptr, dtype_);
}

TensorDispatcher TensorDispatcher::Var(int dim, bool keepdims) const {
    auto var_ptr = tensor_->IVar(dim, keepdims);
    return TensorDispatcher(var_ptr, dtype_);
}

}
#endif