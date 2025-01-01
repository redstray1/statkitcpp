#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "core/config.h"
#include "core/tensor.h"

namespace py = pybind11;

namespace statkitcpp {

PYBIND11_MODULE(statkitcpp, m) {
    m.doc() = R"pbdoc(
        C++ library with various statistical algorithms with Python integration
    )pbdoc";
    py::enum_<data_type>(m, "DataType")
        .value("Float8", data_type::Float8)
        .value("Float16", data_type::Float16)
        .value("Float32", data_type::Float32)
        .value("Float64", data_type::Float64);
    py::class_<Tensor<float>>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<std::vector<uint32_t>, data_type>(), py::arg("shape"), py::arg("dtype")=data_type::Float32)
        .def("__repr__", &Tensor<float>::ToString)
        .def_property_readonly("size", &Tensor<float>::GetSize)
        .def_property("shape", &Tensor<float>::GetShape, &Tensor<float>::SetShape)
        .def_property("requires_grad", &Tensor<float>::GetRequiresGrad, &Tensor<float>::SetRequiresGrad)
        .def("broadcastable_to", &Tensor<float>::BroadcastableTo, py::arg("other"));
    m.def("full", &Tensor<float>::Full, py::arg("shape"), py::arg("value"), py::arg("dtype") = data_type::Float32);
    m.def("zeros", &Tensor<float>::Zeros, py::arg("shape"), py::arg("dtype") = data_type::Float32);
    m.def("ones", &Tensor<float>::Ones, py::arg("shape"), py::arg("dtype") = data_type::Float32);
}
};