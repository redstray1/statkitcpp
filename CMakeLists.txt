cmake_minimum_required(VERSION 3.15...3.27)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/tools/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Statkit LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)


include(tools/cmake/TestSolution.cmake)
include(tools/cmake/BuildFlags.cmake)

add_subdirectory(statkitcpp)
if (CMAKE_TEST_MODE)
    find_package(Catch REQUIRED)
    add_subdirectory(tests)
endif()