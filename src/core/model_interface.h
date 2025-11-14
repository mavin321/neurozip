#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace neurozip {

struct ModelContext;

/// Abstract interface for a compression model.
class ICompressionModel {
public:
    virtual ~ICompressionModel() = default;

    virtual std::unique_ptr<ModelContext> create_context() const = 0;

    /// Given previous byte, update context and produce probability distribution
    /// for next byte as 256 floats (must sum ~1.0).
    virtual void predict_next(
        ModelContext& ctx,
        uint8_t prevByte,
        float* outProbs,
        size_t outSize
    ) const = 0;

    virtual uint32_t model_id() const = 0;
    virtual uint64_t model_hash() const = 0;
};

/// Compress a buffer using a model.
std::vector<uint8_t> compress_buffer(
    const ICompressionModel& model,
    const uint8_t* data,
    size_t size
);

/// Decompress a buffer using a model.
/// On success, returns true and fills outData; on failure, returns false.
bool decompress_buffer(
    const ICompressionModel& model,
    const uint8_t* compressed,
    size_t compressedSize,
    size_t originalSize,
    std::vector<uint8_t>& outData
);

} // namespace neurozip
