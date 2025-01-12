#include "tensor_dispatcher.h"
#include "../errors.h"

namespace statkitcpp {

TensorDispatcher Full(const std::vector<uint32_t>& shape, py::object value, py::str dtype) {
    std::string dtype_str = static_cast<std::string>(dtype);
    if (dtype_str == "float32") {
        Tensor<float> tensor = Tensor<float>::Full(shape, value.cast<float>());
        return TensorDispatcher(std::make_shared<Tensor<float>>(tensor), dtype_str);
    } else if (dtype_str == "float64") {
        Tensor<double> tensor = Tensor<double>::Full(shape, value.cast<double>());
        return TensorDispatcher(std::make_shared<Tensor<double>>(tensor), dtype_str);
    } else {
        throw InvalidDatatypeError{};
    }
}

TensorDispatcher Zeros(const std::vector<uint32_t>& shape, py::str dtype) {
    std::string dtype_str = static_cast<std::string>(dtype);
    if (dtype_str == "float32") {
        Tensor<float> tensor = Tensor<float>::Zeros(shape);
        return TensorDispatcher(std::make_shared<Tensor<float>>(tensor), dtype_str);
    } else if (dtype_str == "float64") {
        Tensor<double> tensor = Tensor<double>::Zeros(shape);
        return TensorDispatcher(std::make_shared<Tensor<double>>(tensor), dtype_str);
    } else {
        throw InvalidDatatypeError{};
    }
}

TensorDispatcher Ones(const std::vector<uint32_t>& shape, py::str dtype) {
    std::string dtype_str = static_cast<std::string>(dtype);
    if (dtype_str == "float32") {
        Tensor<float> tensor = Tensor<float>::Ones(shape);
        return TensorDispatcher(std::make_shared<Tensor<float>>(tensor), dtype_str);
    } else if (dtype_str == "float64") {
        Tensor<double> tensor = Tensor<double>::Ones(shape);
        return TensorDispatcher(std::make_shared<Tensor<double>>(tensor), dtype_str);
    } else {
        throw InvalidDatatypeError{};
    }
}

}