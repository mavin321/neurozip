#include <cassert>
#include <iostream>
#include <vector>
#include "../../src/core/file_format.h"

using namespace neurozip;

int main() {
    std::cout << "[test_file_format] Running...\n";

    FileHeader h;
    h.modelId = 42;
    h.originalSize = 1234;
    h.checksum = 0xdeadbeef;

    std::vector<uint8_t> payload = {1,2,3,4,5};

    write_nzp_file("test_temp.nzp", h, payload);

    FileHeader h2;
    std::vector<uint8_t> p2;
    auto ec = read_nzp_file("test_temp.nzp", h2, p2);

    assert(ec == ErrorCode::Ok);
    assert(h2.modelId == 42);
    assert(h2.originalSize == 1234);
    assert(h2.checksum == 0xdeadbeef);
    assert(p2 == payload);

    std::cout << "[test_file_format] OK\n";
    return 0;
}
