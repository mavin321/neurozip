#include "range_coder.h"

namespace neurozip {

// We use a classic arithmetic coding scheme with [low, high] interval
// on a 32-bit range: [0, 0xFFFFFFFF].
static constexpr uint32_t TOP_MASK   = 0xFF000000u;  // top 8 bits
static constexpr uint32_t FULL_RANGE = 0xFFFFFFFFu;

//
// RangeEncoder
//
RangeEncoder::RangeEncoder()
    : low_(0),
      high_(FULL_RANGE),
      out_()
{
}

void RangeEncoder::output_byte(uint8_t b)
{
    out_.push_back(b);
}

void RangeEncoder::encode_symbol(
    uint32_t cumFreq,
    uint32_t freq,
    uint32_t totalFreq
)
{
    // Guard against invalid parameters
    if (freq == 0 || totalFreq == 0 || cumFreq + freq > totalFreq) {
        // In production you might throw; for tests we just avoid UB.
        return;
    }

    // Current range width
    uint64_t range = (uint64_t)high_ - (uint64_t)low_ + 1u;

    // Update [low, high] to sub-interval for the symbol
    uint64_t lowNew  = (uint64_t)low_ +
                       (range * cumFreq) / totalFreq;
    uint64_t highNew = (uint64_t)low_ +
                       (range * (cumFreq + freq)) / totalFreq - 1u;

    low_  = (uint32_t)lowNew;
    high_ = (uint32_t)highNew;

    // Renormalize: while high and low share the same top byte,
    // shift it out and write it to the stream.
    while ((low_ & TOP_MASK) == (high_ & TOP_MASK)) {
        uint8_t outByte = (uint8_t)(high_ >> 24); // or low_ >> 24
        output_byte(outByte);

        low_  <<= 8;
        high_ <<= 8;
        high_ |= 0xFFu;  // keep interval spanning full 8 bits at the bottom
    }
}

void RangeEncoder::finish()
{
    // After encoding all symbols, we must flush enough bytes to
    // uniquely identify the final [low, high] interval.
    for (int i = 0; i < 4; ++i) {
        uint8_t outByte = (uint8_t)(low_ >> 24);
        output_byte(outByte);
        low_ <<= 8;
    }
}

//
// RangeDecoder
//
RangeDecoder::RangeDecoder(const uint8_t* data, size_t size)
    : low_(0),
      high_(FULL_RANGE),
      code_(0),
      data_(data),
      size_(size),
      pos_(0)
{
    // Initialize 'code_' with first 4 bytes (or fewer, padded with zeros)
    for (int i = 0; i < 4; ++i) {
        code_ = (code_ << 8) | read_byte();
    }
}

uint8_t RangeDecoder::read_byte()
{
    if (pos_ < size_) {
        return data_[pos_++];
    }
    // If we run out of data, pad with zeros (typical arithmetic coder behavior).
    return 0;
}

uint32_t RangeDecoder::get_cum(uint32_t totalFreq) const
{
    if (totalFreq == 0) return 0;

    uint64_t range = (uint64_t)high_ - (uint64_t)low_ + 1u;

    // 'value' is current code position mapped into [0, totalFreq)
    // Formula is the inverse of the encoder's linear mapping.
    uint64_t scaled = ((uint64_t)(code_ - low_ + 1u) * totalFreq - 1u) / range;
    if (scaled >= totalFreq) {
        scaled = totalFreq - 1; // clamp for safety
    }
    return (uint32_t)scaled;
}

void RangeDecoder::decode_symbol(
    uint32_t cumFreq,
    uint32_t freq,
    uint32_t totalFreq
)
{
    if (freq == 0 || totalFreq == 0 || cumFreq + freq > totalFreq) {
        // Invalid; avoid undefined behavior in debug context.
        return;
    }

    uint64_t range = (uint64_t)high_ - (uint64_t)low_ + 1u;

    uint64_t lowNew  = (uint64_t)low_ +
                       (range * cumFreq) / totalFreq;
    uint64_t highNew = (uint64_t)low_ +
                       (range * (cumFreq + freq)) / totalFreq - 1u;

    low_  = (uint32_t)lowNew;
    high_ = (uint32_t)highNew;

    // Renormalize: keep feeding bytes until top byte differs
    while ((low_ & TOP_MASK) == (high_ & TOP_MASK)) {
        low_  <<= 8;
        high_ <<= 8;
        high_ |= 0xFFu;

        code_ = (code_ << 8) | read_byte();
    }
}

} // namespace neurozip
