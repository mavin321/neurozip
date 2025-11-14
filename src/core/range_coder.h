#pragma once

#include <cstdint>
#include <vector>

namespace neurozip {

/// Simple byte-based range coder using a 32-bit range.
/// We encode bytes using cumulative probability tables (0..256).

class RangeEncoder {
public:
    RangeEncoder();

    void encode_symbol(
        uint32_t cumFreq,     // cumulative frequency of symbol
        uint32_t freq,        // frequency of symbol
        uint32_t totalFreq    // total frequency
    );

    void finish();

    const std::vector<uint8_t>& buffer() const { return out_; }

private:
    uint32_t low_;
    uint32_t range_;
    uint32_t cache_;
    uint32_t cacheSize_;
    std::vector<uint8_t> out_;

    void shift();
};

class RangeDecoder {
public:
    RangeDecoder(const uint8_t* data, size_t size);

    /// Given totalFreq and an input value, returns the symbol's cumulative index.
    /// Caller uses this index with its own model/table to find symbol and its freq.
    uint32_t get_cum(uint32_t totalFreq) const;

    /// Update decoder state with symbol frequencies.
    void decode_symbol(
        uint32_t cumFreq,
        uint32_t freq,
        uint32_t totalFreq
    );

    bool eof() const { return pos_ >= size_; }

private:
    uint32_t code_;
    uint32_t range_;
    const uint8_t* data_;
    size_t size_;
    size_t pos_;

    uint8_t read_byte();
};

} // namespace neurozip
