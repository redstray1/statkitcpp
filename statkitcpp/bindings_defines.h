#define AGGREGATION_BINDING(name, func, pyname) \
cls.def(#pyname, &TensorDispatcher::name, py::arg("dim") = -1, py::kw_only(), py::arg("keepdims") = false);

#define BINARY_BINDING(op, name, func, pyname) \
cls.def(#pyname, py::overload_cast<const TensorDispatcher&>(&TensorDispatcher::name, py::const_), py::arg("other")); \
cls.def(#pyname, py::overload_cast<const Scalar&>(&TensorDispatcher::name, py::const_), py::arg("other"));

#define OPERATOR_BINDING(op, name, func, pyname) \
cls.def(py::self op py::self); \
cls.def(py::self op Scalar());

#define OPERATOR_BINDING_NO_OP(op, name, func, pyname) \
cls.def("__" #pyname "__", py::overload_cast<const TensorDispatcher&>(&TensorDispatcher::name, py::const_), py::arg("other")); \
cls.def("__" #pyname "__", py::overload_cast<const Scalar&>(&TensorDispatcher::name, py::const_), py::arg("other"));

#define POINTWISE_BINDING(name, func, pyname) \
cls.def(#pyname, &TensorDispatcher::name);