#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/config.h"
#include "core/tensor.h"

namespace py = pybind11;

namespace statkitcpp {

template <typename T>
void DeclareTensorClass(py::module& module, std::string const & suffix) {
    py::class_<Tensor<T>>(module, ("Tensor" + suffix).c_str())
        .def(py::init<>())
        .def(py::init<std::vector<uint32_t>>(), py::arg("shape"))
        .def("__repr__", &Tensor<T>::ToString)
        .def_property_readonly("size", &Tensor<T>::GetSize)
        .def_property("shape", &Tensor<T>::GetShape, &Tensor<T>::SetShape)
        .def_property("requires_grad", &Tensor<T>::GetRequiresGrad, &Tensor<T>::SetRequiresGrad)
        .def("broadcastable_to", &Tensor<T>::BroadcastableTo, py::arg("other"));
    module.def(("full" + suffix).c_str(), &Tensor<T>::Full, py::arg("shape"), py::arg("value"));
    module.def(("zeros" + suffix).c_str(), &Tensor<T>::Zeros, py::arg("shape"));
    module.def(("ones" + suffix).c_str(), &Tensor<T>::Ones, py::arg("shape"));
}

PYBIND11_MODULE(_statkitcpp, m) {
    m.doc() = R"pbdoc(
        C++ library with various statistical algorithms with Python integration
    )pbdoc";
    py::enum_<data_type>(m, "DataType")
        .value("Float8", data_type::Float8)
        .value("Float16", data_type::Float16)
        .value("Float32", data_type::Float32)
        .value("Float64", data_type::Float64);
    DeclareTensorClass<float>(m, "32");
    DeclareTensorClass<double>(m, "64");
    // m.def("full", &Tensor<float>::Full, py::arg("shape"), py::arg("value"), py::arg("dtype") = data_type::Float32);
    // m.def("zeros", &Tensor<float>::Zeros, py::arg("shape"), py::arg("dtype") = data_type::Float32);
    // m.def("ones", &Tensor<float>::Ones, py::arg("shape"), py::arg("dtype") = data_type::Float32);
}
};