#include "range_coder.h"

namespace neurozip {

// Constants similar to standard range coding implementations.
static constexpr uint32_t TOP = 1u << 24;
static constexpr uint32_t BOT = 1u << 16;

RangeEncoder::RangeEncoder()
    : low_(0),
      range_(~0u),
      cache_(0),
      cacheSize_(0)
{
    out_.clear();
}

void RangeEncoder::shift()
{
    uint8_t byte = static_cast<uint8_t>(low_ >> 24);
    if (cache_ == 0xFF) {
        out_.push_back(cache_ + (byte >> 8));
        while (cacheSize_ > 1) {
            out_.push_back(0xFFu + (byte >> 8));
            --cacheSize_;
        }
        cache_ = byte & 0xFFu;
    } else {
        out_.push_back(cache_);
        while (cacheSize_ > 0) {
            out_.push_back(0xFFu);
            --cacheSize_;
        }
        cache_ = byte;
    }
    low_ = (low_ & 0x00FFFFFFu) << 8;
}

void RangeEncoder::encode_symbol(uint32_t cumFreq, uint32_t freq, uint32_t totalFreq)
{
    // Map cumulative frequencies into [low, low+range)
    range_ /= totalFreq;
    low_ += range_ * cumFreq;
    range_ *= freq;

    while ((low_ ^ (low_ + range_)) < TOP || (range_ < BOT && ((range_ = - static_cast<int32_t>(low_) & (BOT - 1)), true))) {
        shift();
    }
}

void RangeEncoder::finish()
{
    for (int i = 0; i < 4; ++i) {
        shift();
    }
}

RangeDecoder::RangeDecoder(const uint8_t* data, size_t size)
    : code_(0),
      range_(~0u),
      data_(data),
      size_(size),
      pos_(0)
{
    // Initialize "code" with first 4 bytes
    for (int i = 0; i < 4; ++i) {
        code_ = (code_ << 8) | read_byte();
    }
}

uint8_t RangeDecoder::read_byte()
{
    if (pos_ < size_) {
        return data_[pos_++];
    }
    return 0; // implicit zero padding
}

uint32_t RangeDecoder::get_cum(uint32_t totalFreq) const
{
    uint32_t r = range_ / totalFreq;
    uint32_t value = (code_ - 0) / r;
    if (value >= totalFreq) value = totalFreq - 1;
    return value;
}

void RangeDecoder::decode_symbol(uint32_t cumFreq, uint32_t freq, uint32_t totalFreq)
{
    uint32_t r = range_ / totalFreq;
    code_ -= r * cumFreq;
    range_ = r * freq;

    while ((code_ ^ (code_ + range_)) < TOP || (range_ < BOT && ((range_ = -static_cast<int32_t>(code_) & (BOT - 1)), true))) {
        code_ = (code_ << 8) | read_byte();
    }
}

} // namespace neurozip
