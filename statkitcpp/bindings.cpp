#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include "core/datatypes.h"
#include "core/dispatcher/tensor_dispatcher.h"
#include "core/dispatcher/creation_operations.h"

namespace py = pybind11;

namespace statkitcpp {

void DeclareTensorDispatcher(py::module& module) {
    py::class_<TensorDispatcher> cls(module, "Tensor", py::buffer_protocol());
    cls.def(py::init<>());
    cls.def(py::init<std::vector<uint32_t>, py::str, bool>(), py::kw_only(), py::arg("shape"), py::arg("dtype") = "float32", py::arg("requires_grad") = true);
    cls.def("__repr__", &TensorDispatcher::ToString);
    cls.def_property_readonly("size", &TensorDispatcher::GetSize);
    cls.def_property_readonly("ndim", &TensorDispatcher::GetNDim);
    cls.def_property_readonly("itemsize", &TensorDispatcher::GetItemSize);
    cls.def_property_readonly("nbytes", &TensorDispatcher::GetNBytes);
    cls.def_property_readonly("dtype", &TensorDispatcher::GetDType);
    cls.def_property_readonly("strides", &TensorDispatcher::GetStrides);
    cls.def_property("shape", &TensorDispatcher::GetShape, &TensorDispatcher::SetShape);
    cls.def("reshape", &TensorDispatcher::Reshape, py::arg("new_shape"));
    cls.def_property("requires_grad", &TensorDispatcher::GetRequiresGrad, &TensorDispatcher::SetRequiresGrad);
    cls.def("broadcastable_to", &TensorDispatcher::BroadcastableTo, py::arg("other"));

    cls.def_buffer([](TensorDispatcher& td) -> py::buffer_info {
        if (td.GetDType() == "float32") {
            return py::buffer_info(
            td.GetDataPointer(),
            td.GetItemSize(),
            py::format_descriptor<float>::format(),
            td.GetNDim(),
            td.GetShape(),
            td.GetStrides()
        );
        } else {
            return py::buffer_info(
            td.GetDataPointer(),
            td.GetItemSize(),
            py::format_descriptor<double>::format(),
            td.GetNDim(),
            td.GetShape(),
            td.GetStrides()
        );
        }
        
    });

    cls.def(py::init<py::buffer>());

    module.def("full", &Full, py::kw_only(), py::arg("shape"), py::arg("value"), py::arg("dtype") = "float32");
    module.def("zeros", &Zeros, py::kw_only(), py::arg("shape"), py::arg("dtype") = "float32");
    module.def("ones", &Ones, py::kw_only(), py::arg("shape"), py::arg("dtype") = "float32");
}

PYBIND11_MODULE(_statkitcpp, m) {
    m.doc() = R"pbdoc(
        C++ library with various statistical algorithms with Python integration
    )pbdoc";

    py::class_<Float32>(m, "Float32");
    py::class_<Float64>(m, "Float64");
    // DeclareTensorClass<float>(m, "32");
    // DeclareTensorClass<double>(m, "64");
    DeclareTensorDispatcher(m);
}
};