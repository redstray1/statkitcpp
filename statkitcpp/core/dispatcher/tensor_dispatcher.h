#ifndef TENSOR_BINDING_HEADER_H
#define TENSOR_BINDING_HEADER_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <memory>
#include "../_tensor/shape.h"
#include "../_tensor/ScalarType.h"
#include "dispatcher_defines.h"
#include "../_tensor/Tensor.h"
#include "../errors.h"

namespace py = pybind11;

namespace statkitcpp {

class TensorDispatcher {
private:
    Tensor tensor_;
public:
    TensorDispatcher();
    TensorDispatcher(const std::vector<size_t>& shape, ScalarType dtype, bool requires_grad = false);
    TensorDispatcher(const TensorDispatcher& other) = default;
    TensorDispatcher(TensorDispatcher&& other) = default;
    // TensorDispatcher(std::shared_ptr<Tensor> tensor)
    //     : tensor_(std::move(tensor)) {}
    // TensorDispatcher(const Tensor& tensor)
    //     : tensor_(std::make_shared<Tensor>(tensor)) {}
    // TensorDispatcher(Tensor&& tensor)
    //     : tensor_(std::make_shared<Tensor>(tensor)) {}

    // TensorDispatcher(Tensor* tensor) { tensor_ = tensor; }
    TensorDispatcher(const Tensor& tensor) { tensor_ = tensor; }
    TensorDispatcher(Tensor&& tensor) {tensor_ = std::move(tensor); }

    TensorDispatcher(py::buffer b);
    // void ClearData() { 
    //     if (tensor_) { delete tensor_; } 
    //     tensor_ = nullptr;
    // }
    ~TensorDispatcher() { 
        // ClearData();
    }

    TensorDispatcher& operator=(const TensorDispatcher& other) = default;
    TensorDispatcher& operator=(TensorDispatcher&& other) = default;

    std::string ToString() const;
    bool BroadcastableTo(const TensorDispatcher& other);

    void Backward(std::optional<TensorDispatcher> grad_output = std::nullopt,
                  std::optional<TensorDispatcher> output = std::nullopt,
                  bool retain_graph = false);

    std::vector<size_t> GetShape() const;
    void SetShape(const std::vector<size_t>& shape);
    TensorDispatcher Reshape(const std::vector<size_t>& shape);

    std::vector<size_t> GetStrides() const;

    size_t GetSize() const;
    
    size_t GetNDim() const;

    void SetRequiresGrad(bool requires_grad);
    bool GetRequiresGrad() const;

    std::optional<TensorDispatcher> GetGrad() const;
    auto GetDataType() const;
    void* GetDataPointer();
    ScalarType GetDType() const;
    size_t GetItemSize() const;
    size_t GetNBytes() const;

    TENSOR_AGGREGATION_METHODS(DISPATCHER_AGGREGATION_DECLARATIONS)

    TENSOR_BINARY_DECLARATIONS_WITH_OP(DISPATCHER_OPERATOR_METHODS_DECLARATIONS)
    TENSOR_BINARY_DECLARATIONS_WITHOUT_OP(DISPATCHER_OPERATOR_METHODS_DECLARATIONS)

    TENSOR_LINALG_OPERATIONS_WITH_OP(DISPATCHER_LINALG_METHODS_DECLARATIONS)
    TENSOR_LINALG_OPERATIONS_WITHOUT_OP(DISPATCHER_LINALG_METHODS_DECLARATIONS)

    TENSOR_BINARY_DECLARATIONS_WITH_OP(DISPATCHER_OPERATOR_DECLARATIONS)
    TENSOR_POINTWISE_METHODS(DISPATCHER_POINTWISE_DECLARATIONS)
};  

TensorDispatcher::TensorDispatcher() {
    // tensor_ = std::make_shared<Tensor>();
    tensor_ = Tensor();
}

TensorDispatcher::TensorDispatcher(const std::vector<size_t>& shape, ScalarType dtype, bool requires_grad) {
    // tensor_ = std::make_shared<Tensor>(shape, dtype, requires_grad);
    tensor_ = Tensor(shape, dtype, requires_grad);
}

ScalarType NumpyTypeToScalarType(py::buffer_info& info) {
    if (info.item_type_is_equivalent_to<int8_t>()) {
        return ScalarType::Char;
    } else if (info.item_type_is_equivalent_to<int16_t>()) {
        return ScalarType::Short;
    } else if (info.item_type_is_equivalent_to<int>()) {
        return ScalarType::Int;
    } else if (info.item_type_is_equivalent_to<int64_t>()) {
        return ScalarType::Long;
    } else if (info.item_type_is_equivalent_to<float>()) {
        return ScalarType::Float;
    } else if (info.item_type_is_equivalent_to<double>()) {
        return ScalarType::Double;
    } else if (info.item_type_is_equivalent_to<bool>()) {
        return ScalarType::Bool;
    } else {
        throw InvalidDatatypeError{};
    }
}

TensorDispatcher::TensorDispatcher(py::buffer b) {
    py::buffer_info info = b.request();
    std::vector<size_t> shape;
    std::transform(info.shape.begin(), info.shape.end(), std::back_inserter(shape), [](const int value)
    {
        return static_cast<size_t>(value);
    });
    // tensor_ = std::make_shared<Tensor>(info.ptr, shape, NumpyTypeToScalarType(info), false);
    tensor_ = Tensor(info.ptr, shape, NumpyTypeToScalarType(info), false);
}

std::string TensorDispatcher::ToString() const {
    return tensor_.ToString();
}

void TensorDispatcher::Backward(std::optional<TensorDispatcher> grad_output,
                                std::optional<TensorDispatcher> output,
                                bool retain_graph) {
    std::optional<Tensor> grad_o = std::nullopt;
    std::optional<Tensor> o = std::nullopt;
    if (grad_output.has_value()) {
        grad_o = (grad_output.value().tensor_);
    }
    if (output.has_value()) {
        o = (output.value().tensor_);
    }
    tensor_.Backward(grad_o, o, retain_graph);
}

bool TensorDispatcher::BroadcastableTo(const TensorDispatcher& other) {
    return IsBroadcastable(tensor_.GetShape(), other.GetShape());
}

std::vector<size_t> TensorDispatcher::GetShape() const {
    return tensor_.GetShape();
}

void TensorDispatcher::SetShape(const std::vector<size_t>& shape) {
    tensor_.SetShape(shape);
}

TensorDispatcher TensorDispatcher::Reshape(const std::vector<size_t>& shape) {
    auto var_ptr = tensor_.Reshape(shape);
    return TensorDispatcher(var_ptr);
} 

std::vector<size_t> TensorDispatcher::GetStrides() const {
    return tensor_.GetStrides();
}

size_t TensorDispatcher::GetSize() const {
    return tensor_.GetSize();
}

size_t TensorDispatcher::GetNDim() const {
    return tensor_.GetNDim();
}

void TensorDispatcher::SetRequiresGrad(bool requires_grad) {
    tensor_.SetRequiresGrad(requires_grad);
}

bool TensorDispatcher::GetRequiresGrad() const {
    return tensor_.GetRequiresGrad();
}

void* TensorDispatcher::GetDataPointer() {
    return tensor_.GetDataPointer();
}

ScalarType TensorDispatcher::GetDType() const {
    return tensor_.GetDType();
}

size_t TensorDispatcher::GetItemSize() const {
    return tensor_.GetItemSize();
}

size_t TensorDispatcher::GetNBytes() const {
    return tensor_.GetNBytes();
}

std::optional<TensorDispatcher> TensorDispatcher::GetGrad() const {
    // if (tensor_.grad == nullptr) {
    //     return std::nullopt;
    // }
    // return *tensor_.grad;
    return tensor_.GetGrad();
}

TENSOR_AGGREGATION_METHODS(DISPATCHER_AGGREGATION_DEFINITIONS)


TENSOR_BINARY_DECLARATIONS_WITH_OP(DISPATCHER_OPERATOR_METHODS_DEFINITIONS)
TENSOR_BINARY_DECLARATIONS_WITHOUT_OP(DISPATCHER_OPERATOR_METHODS_DEFINITIONS)

TENSOR_POINTWISE_METHODS(DISPATCHER_POINTWISE_DEFINITIONS)
TENSOR_LINALG_OPERATIONS_WITH_OP(DISPATCHER_LINALG_METHODS_DEFINITIONS)
TENSOR_LINALG_OPERATIONS_WITHOUT_OP(DISPATCHER_LINALG_METHODS_DEFINITIONS)

TENSOR_BINARY_DECLARATIONS_WITH_OP(DISPATCHER_OPERATOR_DEFINITIONS)

}
#endif