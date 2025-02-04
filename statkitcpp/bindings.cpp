#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include "./ScalarType.h"
#include "Scalar.h"
#include "core/datatypes.h"
#include "core/dispatcher/tensor_dispatcher.h"
#include "core/dispatcher/creation_operations.h"
#include "bindings_defines.h"
#include "core/_tensor/tensor_defines.h"
#include "errors.h"
//#include "core/_tensor/tensor.h"
//#include "core/_tensor/TensorCreationOps.h"

namespace py = pybind11;

namespace statkitcpp {

void DeclareTensor(py::module& module) {
    py::class_<TensorDispatcher> cls(module, "Tensor", py::buffer_protocol());
    cls.def(py::init<>());
    cls.def(py::init<std::vector<size_t>, ScalarType, bool>(), py::kw_only(), py::arg("shape"), py::arg("dtype") = ScalarType::Float, py::arg("requires_grad") = true);
    cls.def("__repr__", &TensorDispatcher::ToString);
    cls.def_property_readonly("size", &TensorDispatcher::GetSize);
    cls.def_property_readonly("ndim", &TensorDispatcher::GetNDim);
    cls.def_property_readonly("itemsize", &TensorDispatcher::GetItemSize);
    cls.def_property_readonly("nbytes", &TensorDispatcher::GetNBytes);
    cls.def_property_readonly("dtype", &TensorDispatcher::GetDType);
    cls.def_property_readonly("strides", &TensorDispatcher::GetStrides);
    cls.def_property_readonly("grad", &TensorDispatcher::GetGrad);
    cls.def_property("shape", &TensorDispatcher::GetShape, &TensorDispatcher::SetShape);
    cls.def_property("requires_grad", &TensorDispatcher::GetRequiresGrad, &TensorDispatcher::SetRequiresGrad);
    cls.def("broadcastable_to", &TensorDispatcher::BroadcastableTo, py::arg("other"));
    
    TENSOR_AGGREGATION_METHODS(AGGREGATION_BINDING)

    TENSOR_BINARY_DECLARATIONS_WITH_OP(BINARY_BINDING)
    TENSOR_BINARY_DECLARATIONS_WITHOUT_OP(BINARY_BINDING)

    TENSOR_BINARY_DECLARATIONS_WITH_OP(OPERATOR_BINDING)

    TENSOR_BINARY_DECLARATIONS_WITHOUT_OP(OPERATOR_BINDING_NO_OP)

    TENSOR_LINALG_OPERATIONS_WITH_OP(LINALG_OPERATOR_BINDING)
    TENSOR_LINALG_OPERATIONS_WITH_OP(LINALG_METHOD_BINDING)
    TENSOR_LINALG_OPERATIONS_WITHOUT_OP(LINALG_METHOD_BINDING)

    TENSOR_POINTWISE_METHODS(POINTWISE_BINDING)

    //Tensor operations
    cls.def("reshape", &TensorDispatcher::Reshape, py::arg("shape"));

    //Backward method
    cls.def("backward", &TensorDispatcher::Backward, py::arg("grad_output") = py::none(), py::arg("output") = py::none(), py::arg("retain_graph") = false);

    #define DEFINE_BUFFER_INFO(T, name) \
    case ScalarType::name: {\
        std::vector<size_t> strides = td.GetStrides(); \
        for (auto& x : strides) { \
            x *= td.GetItemSize(); \
        } \
        return py::buffer_info( \
            td.GetDataPointer(), \
            td.GetItemSize(), \
            py::format_descriptor<T>::format(), \
            td.GetNDim(), \
            td.GetShape(), \
            strides \
        ); \
    }

    cls.def_buffer([](TensorDispatcher& td) -> py::buffer_info {
        switch(td.GetDType()) {
            SCALAR_TYPES(DEFINE_BUFFER_INFO)
            default:
                throw InvalidDatatypeError{};
        }
        }
    );
    #undef DEFINE_BUFFER_INFO

    cls.def(py::init<py::buffer>());

    module.def("full", &FullPy, py::arg("shape"), py::arg("value"),py::kw_only(),  py::arg("dtype") = ScalarType::Float);
    module.def("zeros", &ZerosPy, py::arg("shape"),py::kw_only(),  py::arg("dtype") = ScalarType::Float);
    module.def("ones", &OnesPy, py::arg("shape"),py::kw_only(),  py::arg("dtype") = ScalarType::Float);
    module.def("arange", &ArangePy, py::arg("start"), py::arg("end"), py::arg("step") = Scalar(1),py::kw_only(),  py::arg("dtype") = ScalarType::Float);
}

PYBIND11_MODULE(_statkitcpp, m) {
    m.doc() = R"pbdoc(
        C++ library with various statistical algorithms with Python integration
    )pbdoc";

    py::enum_<ScalarType>(m, "ScalarType")
        .value("int8", kInt8)
        .value("int16", kInt16)
        .value("int32", kInt32)
        .value("int64", kInt64)
        .value("bool", ScalarType::Bool)
        .value("float32", kFloat32)
        .value("float64", kFloat64)
        .export_values();

    py::class_<Scalar> scalar(m, "Scalar");
    #define SCALAR_CTOR(T, name) \
    scalar.def(py::init<T>());

    SCALAR_TYPES(SCALAR_CTOR)
    #undef SCALAR_CTOR
    #define SCALAR_IMPLICIT_CONVERT(T, name) \
    py::implicitly_convertible<T, Scalar>();

    SCALAR_TYPES(SCALAR_IMPLICIT_CONVERT)
    #undef SCALAR_IMPLICIT_CONVERT

    // DeclareTensorClass<float>(m, "32");
    // DeclareTensorClass<double>(m, "64");
    DeclareTensor(m);
}
};