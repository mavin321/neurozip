# Integration Tests

add_executable(test_roundtrip test_roundtrip.cpp)
target_link_libraries(test_roundtrip PRIVATE neurozip_core)
add_test(NAME TestRoundtrip COMMAND test_roundtrip)
