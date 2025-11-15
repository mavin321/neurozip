# Unit Tests

# TestCodec
add_executable(test_codec test_codec.cpp)
target_link_libraries(test_codec PRIVATE neurozip_core)
add_test(NAME TestCodec COMMAND test_codec)

# TestModelInterface
add_executable(test_model_interface test_model_interface.cpp)
target_link_libraries(test_model_interface PRIVATE neurozip_core)
add_test(NAME TestModelInterface COMMAND test_model_interface)
