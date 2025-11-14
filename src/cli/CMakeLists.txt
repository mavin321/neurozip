add_executable(neurozip cli_neurozip.cpp)
target_link_libraries(neurozip PRIVATE neurozip)

add_executable(neurounzip cli_neurounzip.cpp)
target_link_libraries(neurounzip PRIVATE neurozip)

add_executable(neurozip-inspect cli_inspect.cpp)
target_link_libraries(neurozip-inspect PRIVATE neurozip)
