#pragma once

#include <cstdint>
#include <vector>

namespace neurozip {

/// Simple 32-bit arithmetic coder over bytes.
/// Uses cumulative frequencies in [0, totalFreq].

class RangeEncoder {
public:
    RangeEncoder();

    // Encode symbol with cumulative frequency 'cumFreq' and width 'freq'
    // where totalFreq = sum of all symbol frequencies.
    void encode_symbol(
        uint32_t cumFreq,
        uint32_t freq,
        uint32_t totalFreq
    );

    // Finalize the stream (flush remaining state).
    void finish();

    const std::vector<uint8_t>& buffer() const { return out_; }

private:
    uint32_t low_;     // low end of current interval
    uint32_t high_;    // high end of current interval
    std::vector<uint8_t> out_;

    void output_byte(uint8_t b);
};

class RangeDecoder {
public:
    RangeDecoder(const uint8_t* data, size_t size);

    /// Return the cumulative index in [0, totalFreq) corresponding
    /// to the current code position.
    uint32_t get_cum(uint32_t totalFreq) const;

    /// Advance the decoder state by consuming the symbol with
    /// [cumFreq, cumFreq + freq) in the cumulative distribution.
    void decode_symbol(
        uint32_t cumFreq,
        uint32_t freq,
        uint32_t totalFreq
    );

    bool eof() const { return pos_ >= size_; }

private:
    uint32_t low_;
    uint32_t high_;
    uint32_t code_;

    const uint8_t* data_;
    size_t size_;
    size_t pos_;

    uint8_t read_byte();
};

} // namespace neurozip
