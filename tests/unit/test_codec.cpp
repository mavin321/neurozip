#include <cassert>
#include <vector>
#include <iostream>
#include "../../src/core/range_coder.h"

using namespace neurozip;

int main() {
    std::cout << "[test_codec] Running tests...\n";

    // Tiny uniform distribution for sanity
    float probs[256];
    for (int i = 0; i < 256; i++) probs[i] = 1.0f / 256.0f;

    // fake cumfreq
    uint32_t cum[257];
    for (int i = 0; i <= 256; i++) cum[i] = i;
    uint32_t total = 256;

    // encode simple sequence
    RangeEncoder enc;
    for (int b : {1, 2, 3, 4, 5}) {
        enc.encode_symbol(cum[b], 1, total);
    }
    enc.finish();

    auto buf = enc.buffer();

    // decode
    RangeDecoder dec(buf.data(), buf.size());

    for (int expected = 1; expected <= 5; expected++) {
        uint32_t v = dec.get_cum(total);
        assert(v == (uint32_t)expected);
        dec.decode_symbol(expected, 1, total);
    }

    std::cout << "[test_codec] OK\n";
    return 0;
}
