set(NEUROZIP_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_library(neurozip STATIC
    core/file_format.cpp
    core/range_coder.cpp
    core/model_interface.cpp
    models/tiny_lstm.cpp
    api/neurozip_c.cpp
    api/neurozip_cpp.cpp
)

target_include_directories(neurozip
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${NEUROZIP_SRC_ROOT}
)

target_compile_definitions(neurozip PRIVATE -DNEUROZIP_VERSION="1.0.0")
