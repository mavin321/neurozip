cmake_minimum_required(VERSION 3.16)
project(neurozip VERSION 1.0.0 DESCRIPTION "AI-powered text compression engine")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NEUROZIP_BUILD_TESTS "Build unit tests" ON)
option(NEUROZIP_BUILD_PYTHON "Build Python bindings" ON)

# Core library
add_subdirectory(src)



# CLI tools
add_subdirectory(src/cli)
