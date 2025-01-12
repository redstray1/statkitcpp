#ifndef TENSOR_BINDING_HEADER_H
#define TENSOR_BINDING_HEADER_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <memory>
#include "../_tensor/tensor.h"
#include "../_tensor/shape.h"
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
    TensorDispatcher(std::shared_ptr<Variable> tensor) : tensor_(std::move(tensor)) {}
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

    //auto GetData();
    std::string GetDType() const;
    uint32_t GetItemSize() const;
    uint32_t GetNBytes() const;
};  

TensorDispatcher::TensorDispatcher() {
    dtype_ = "float32";
    tensor_ = std::make_shared<Tensor<float>>();
}

TensorDispatcher::TensorDispatcher(const std::vector<uint32_t>& shape, py::str dtype, bool requires_grad) {
    std::string dtype_str = static_cast<std::string>(dtype);
    if (dtype_str == "float32") {
        //tensor_ = new Tensor<float>(shape, requires_grad);
        tensor_ = std::make_shared<Tensor<float>>(shape, requires_grad);
        dtype_ = dtype;
    } else if (dtype_str == "float64") {
        tensor_ = std::make_shared<Tensor<double>>(shape, requires_grad);
        dtype_ = dtype;
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

std::string TensorDispatcher::GetDType() const {
    return dtype_;
}

uint32_t TensorDispatcher::GetItemSize() const {
    return tensor_->GetItemSize();
}

uint32_t TensorDispatcher::GetNBytes() const {
    return tensor_->GetNBytes();
}

}
#endif