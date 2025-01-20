#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "./ScalarType.h"
#include "Scalar.h"
#include "core/datatypes.h"
#include "core/dispatcher/tensor_dispatcher.h"
#include "core/dispatcher/creation_operations.h"
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
    cls.def_property("shape", &TensorDispatcher::GetShape, &TensorDispatcher::SetShape);
    cls.def_property("requires_grad", &TensorDispatcher::GetRequiresGrad, &TensorDispatcher::SetRequiresGrad);
    cls.def("broadcastable_to", &TensorDispatcher::BroadcastableTo, py::arg("other"));
    
    //Aggregation functions
    cls.def("sum", &TensorDispatcher::Sum, py::arg("dim") = -1, py::kw_only(), py::arg("keepdims") = false);
    cls.def("prod", &TensorDispatcher::Prod, py::arg("dim") = -1, py::kw_only(), py::arg("keepdims") = false);
    cls.def("mean", &TensorDispatcher::Mean, py::arg("dim") = -1, py::kw_only(), py::arg("keepdims") = false);
    cls.def("var", &TensorDispatcher::Var, py::arg("dim") = -1, py::kw_only(), py::arg("keepdims") = false);
    
    //Binary operations
    cls.def("add", &TensorDispatcher::Add, py::arg("other"), py::arg("alpha") = Scalar(1));
    cls.def("sub", &TensorDispatcher::Sub, py::arg("other"), py::arg("alpha") = Scalar(1));
    cls.def("mul", &TensorDispatcher::Mul, py::arg("other"));
    cls.def("div", &TensorDispatcher::Div, py::arg("other"));
    cls.def("pow", &TensorDispatcher::Pow, py::arg("other"));

    //Pointwise operations
    cls.def("neg", &TensorDispatcher::Neg);
    cls.def("exp", &TensorDispatcher::Exp);
    cls.def("log", &TensorDispatcher::Log);
    cls.def("sqrt", &TensorDispatcher::Sqrt);

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

    module.def("full", &FullPy, py::kw_only(), py::arg("shape"), py::arg("value"), py::arg("dtype") = ScalarType::Float);
    module.def("zeros", &ZerosPy, py::kw_only(), py::arg("shape"), py::arg("dtype") = ScalarType::Float);
    module.def("ones", &OnesPy, py::kw_only(), py::arg("shape"), py::arg("dtype") = ScalarType::Float);
    module.def("arange", &ArangePy, py::kw_only(), py::arg("start"), py::arg("end"), py::arg("step") = Scalar(1), py::arg("dtype") = ScalarType::Float);
}

// void DeclareTensorDispatcher(py::module& module) {
//     py::class_<TensorDispatcher> cls(module, "Tensor", py::buffer_protocol());
//     cls.def(py::init<>());
//     cls.def(py::init<std::vector<uint32_t>, py::str, bool>(), py::kw_only(), py::arg("shape"), py::arg("dtype") = "float32", py::arg("requires_grad") = true);
//     cls.def("__repr__", &TensorDispatcher::ToString);
//     cls.def_property_readonly("size", &TensorDispatcher::GetSize);
//     cls.def_property_readonly("ndim", &TensorDispatcher::GetNDim);
//     cls.def_property_readonly("itemsize", &TensorDispatcher::GetItemSize);
//     cls.def_property_readonly("nbytes", &TensorDispatcher::GetNBytes);
//     cls.def_property_readonly("dtype", &TensorDispatcher::GetDType);
//     cls.def_property_readonly("strides", &TensorDispatcher::GetStrides);
//     cls.def_property("shape", &TensorDispatcher::GetShape, &TensorDispatcher::SetShape);
//     cls.def("reshape", &TensorDispatcher::Reshape, py::arg("new_shape"));
//     cls.def_property("requires_grad", &TensorDispatcher::GetRequiresGrad, &TensorDispatcher::SetRequiresGrad);
//     cls.def("broadcastable_to", &TensorDispatcher::BroadcastableTo, py::arg("other"));
//     cls.def("sum", &TensorDispatcher::Sum, py::arg("dim") = -1, py::kw_only(), py::arg("keepdims") = false);
//     cls.def("mean", &TensorDispatcher::Mean, py::arg("dim") = -1, py::kw_only(), py::arg("keepdims") = false);
//     cls.def("var", &TensorDispatcher::Var, py::arg("dim") = -1, py::kw_only(), py::arg("keepdims") = false);

//     cls.def_buffer([](TensorDispatcher& td) -> py::buffer_info {
//         if (td.GetDType() == "float32") {
//             return py::buffer_info(
//             td.GetDataPointer(),
//             td.GetItemSize(),
//             py::format_descriptor<float>::format(),
//             td.GetNDim(),
//             td.GetShape(),
//             td.GetStrides()
//         );
//         } else {
//             return py::buffer_info(
//             td.GetDataPointer(),
//             td.GetItemSize(),
//             py::format_descriptor<double>::format(),
//             td.GetNDim(),
//             td.GetShape(),
//             td.GetStrides()
//         );
//         }
        
//     });

//     cls.def(py::init<py::buffer>());

//     module.def("full", &Full, py::kw_only(), py::arg("shape"), py::arg("value"), py::arg("dtype") = "float32");
//     module.def("zeros", &Zeros, py::kw_only(), py::arg("shape"), py::arg("dtype") = "float32");
//     module.def("ones", &Ones, py::kw_only(), py::arg("shape"), py::arg("dtype") = "float32");
// }

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

    // DeclareTensorClass<float>(m, "32");
    // DeclareTensorClass<double>(m, "64");
    DeclareTensor(m);
}
};