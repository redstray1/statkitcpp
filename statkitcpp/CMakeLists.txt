add_library(libcore STATIC
    core/_tensor/Tensor.cpp
    core/_tensor/TensorImpl.cpp
    core/_tensor/TensorOperations.cpp
    core/_tensor/shape.cpp
    core/_autograd/autograd.cpp
    core/_autograd/Node.cpp
    core/_autograd/TensorOperations.cpp
    core/_autograd/BinaryOperations.cpp
    core/_autograd/PointwiseOperations.cpp
    core/_autograd/AggregationOperations.cpp
    core/_tensor/tensor_creation_ops.cpp
    core/_tensor/TensorCreationOps.cpp
    core/_tensor/ScalarType.cpp
    core/_tensor/Storage.cpp
    core/_tensor/Scalar.cpp
    core/_tensor/operations/Operations.cpp
    core/_tensor/operations/PointwiseOperations.cpp
    core/_tensor/operations/TensorOperations.cpp
)
set_target_properties(libcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils)
target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/_autograd)
target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/_tensor/operations)
target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/_tensor)
target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/dispatcher)
target_include_directories(libcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)

pybind11_add_module(_statkitcpp bindings.cpp)
target_link_libraries(_statkitcpp PRIVATE libcore)

install(TARGETS _statkitcpp DESTINATION .)