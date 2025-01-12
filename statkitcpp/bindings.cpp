#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include "core/datatypes.h"
#include "core/_tensor/tensor.h"
#include "core/dispatcher/tensor_dispatcher.h"
#include "core/dispatcher/creation_operations.h"

namespace py = pybind11;

namespace statkitcpp {

template <typename T>
void DeclareTensorClass(py::module& module, std::string const & suffix) {
    py::class_<Tensor<T>>(module, ("Tensor" + suffix).c_str())
        .def(py::init<>())
        .def(py::init<std::vector<uint32_t>, bool>(), py::kw_only(), py::arg("shape"), py::arg("requires_grad") = true)
        .def("__repr__", &Tensor<T>::ToString)
        .def_property_readonly("size", &Tensor<T>::GetSize)
        .def_property_readonly("ndim", &Tensor<T>::GetNDim)
        .def_property_readonly("itemsize", &Tensor<T>::GetItemSize)
        .def_property_readonly("nbytes", &Tensor<T>::GetNBytes)
        .def_property("shape", &Tensor<T>::GetShape, &Tensor<T>::SetShape)
        .def_property("requires_grad", &Tensor<T>::GetRequiresGrad, &Tensor<T>::SetRequiresGrad)
        .def("broadcastable_to", &Tensor<T>::BroadcastableTo, py::arg("other"));
    module.def(("full" + suffix).c_str(), &Tensor<T>::Full, py::arg("shape"), py::arg("value"));
    module.def(("zeros" + suffix).c_str(), &Tensor<T>::Zeros, py::arg("shape"));
    module.def(("ones" + suffix).c_str(), &Tensor<T>::Ones, py::arg("shape"));
}

void DeclareTensorDispatcher(py::module& module) {
    py::class_<TensorDispatcher> cls(module, "Tensor");
    cls.def(py::init<>());
    cls.def(py::init<std::vector<uint32_t>, py::str, bool>(), py::kw_only(), py::arg("shape"), py::arg("dtype") = "float32", py::arg("requires_grad") = true);
    cls.def("__repr__", &TensorDispatcher::ToString);
    cls.def_property_readonly("size", &TensorDispatcher::GetSize);
    cls.def_property_readonly("ndim", &TensorDispatcher::GetNDim);
    cls.def_property_readonly("itemsize", &TensorDispatcher::GetItemSize);
    cls.def_property_readonly("nbytes", &TensorDispatcher::GetNBytes);
    cls.def_property_readonly("dtype", &TensorDispatcher::GetDType);
    cls.def_property("shape", &TensorDispatcher::GetShape, &TensorDispatcher::SetShape);
    cls.def("reshape", &TensorDispatcher::Reshape, py::arg("new_shape"));
    cls.def_property("requires_grad", &TensorDispatcher::GetRequiresGrad, &TensorDispatcher::SetRequiresGrad);
    cls.def("broadcastable_to", &TensorDispatcher::BroadcastableTo, py::arg("other"));

    module.def("full", &Full, py::kw_only(), py::arg("shape"), py::arg("value"), py::arg("dtype") = "float32");
    module.def("zeros", &Zeros, py::kw_only(), py::arg("shape"), py::arg("dtype") = "float32");
    module.def("ones", &Ones, py::kw_only(), py::arg("shape"), py::arg("dtype") = "float32");
}

PYBIND11_MODULE(_statkitcpp, m) {
    m.doc() = R"pbdoc(
        C++ library with various statistical algorithms with Python integration
    )pbdoc";
    // py::enum_<data_type>(m, "DataType")
    //     .value("Float8", data_type::Float8)
    //     .value("Float16", data_type::Float16)
    //     .value("Float32", data_type::Float32)
    //     .value("Float64", data_type::Float64);

    py::class_<Float32>(m, "Float32");
    py::class_<Float64>(m, "Float64");
    // DeclareTensorClass<float>(m, "32");
    // DeclareTensorClass<double>(m, "64");
    DeclareTensorDispatcher(m);
}
};