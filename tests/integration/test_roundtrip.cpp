#include <cassert>
#include <iostream>
#include <fstream>
#include "../../src/api/neurozip_cpp.h"

using namespace neurozip;

int main() {
    std::cout << "[test_roundtrip] Running...\n";

    const char* modelPath = "tiny_lstm.bin";
    Model m(modelPath);
    assert(m.valid());

    // Write input
    const char* text = "This is a test of the neurozip roundtrip system.";
    std::ofstream("rt_input.txt") << text;

    auto c = compress_file("rt_input.txt", "rt_output.nzp", m);
    assert(c == NZP_OK);

    auto d = decompress_file("rt_output.nzp", "rt_restored.txt", m);
    assert(d == NZP_OK);

    std::ifstream ifs("rt_restored.txt");
    std::string restored((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    assert(restored == text);

    std::cout << "[test_roundtrip] OK\n";
    return 0;
}
