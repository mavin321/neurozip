#include "model_interface.h"
#include "range_coder.h"

#include <algorithm>
#include <cmath>

namespace neurozip {

/// Convert probability distribution (256 floats) to cumulative frequencies.
static void probs_to_cumfreq(
    const float* probs,
    uint32_t* cum,
    uint32_t& total
) {
    // Simple scaling to 16-bit frequencies
    constexpr uint32_t SCALE = 1u << 15; // total around 32768
    total = 0;
    cum[0] = 0;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t f = static_cast<uint32_t>(probs[i] * SCALE);
        if (f == 0) f = 1; // avoid zero frequencies
        cum[i + 1] = cum[i] + f;
    }
    total = cum[256];
    if (total == 0) {
        // fallback to uniform
        for (uint32_t i = 0; i <= 256; ++i) {
            cum[i] = i;
        }
        total = 256;
    }
}

std::vector<uint8_t> compress_buffer(
    const ICompressionModel& model,
    const uint8_t* data,
    size_t size
) {
    RangeEncoder encoder;
    auto ctx = model.create_context();

    float probs[256];
    uint32_t cum[257];
    uint32_t total = 0;

    uint8_t prev = 0; // BOS symbol

    for (size_t i = 0; i < size; ++i) {
        model.predict_next(*ctx, prev, probs, 256);
        probs_to_cumfreq(probs, cum, total);

        uint8_t sym = data[i];

        uint32_t cumFreq = cum[sym];
        uint32_t freq = cum[sym + 1] - cum[sym];

        encoder.encode_symbol(cumFreq, freq, total);
        prev = sym;
    }

    // Encode EOF as 256? We just finish; length is known externally.
    encoder.finish();
    return encoder.buffer();
}

bool decompress_buffer(
    const ICompressionModel& model,
    const uint8_t* compressed,
    size_t compressedSize,
    size_t originalSize,
    std::vector<uint8_t>& outData
) {
    outData.clear();
    outData.reserve(originalSize);

    RangeDecoder decoder(compressed, compressedSize);
    auto ctx = model.create_context();

    float probs[256];
    uint32_t cum[257];
    uint32_t total = 0;

    uint8_t prev = 0;

    for (size_t i = 0; i < originalSize; ++i) {
        model.predict_next(*ctx, prev, probs, 256);
        probs_to_cumfreq(probs, cum, total);

        uint32_t value = decoder.get_cum(total);

        // Find symbol by binary search on cum[]
        uint32_t lo = 0, hi = 256;
        while (lo + 1 < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (cum[mid] > value)
                hi = mid;
            else
                lo = mid;
        }
        uint32_t sym = lo;

        uint32_t cumFreq = cum[sym];
        uint32_t freq = cum[sym + 1] - cum[sym];

        decoder.decode_symbol(cumFreq, freq, total);

        outData.push_back(static_cast<uint8_t>(sym));
        prev = static_cast<uint8_t>(sym);
    }

    return true;
}

} // namespace neurozip
