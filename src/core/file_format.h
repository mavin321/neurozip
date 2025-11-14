#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace neurozip {

constexpr uint32_t NZP_MAGIC = 0x31505A4E; // "NZP1" little-endian
constexpr uint8_t  NZP_FORMAT_VERSION = 1;

enum class ErrorCode {
    Ok = 0,
    IoError,
    InvalidFormat,
    UnsupportedVersion,
    ModelMismatch,
    CorruptData,
    InternalError
};

struct FileHeader {
    uint32_t magic;         // NZP_MAGIC
    uint8_t  formatVersion; // NZP_FORMAT_VERSION
    uint32_t modelId;
    uint8_t  flags;
    uint64_t originalSize;
    uint32_t checksum;      // CRC32 of original data
    uint64_t modelHash;
    uint64_t reserved;      // for future use

    FileHeader();
};

/// Compute CRC32 of a buffer (simple implementation).
uint32_t crc32(const uint8_t* data, size_t len, uint32_t seed = 0);

/// Write header and payload to a file.
ErrorCode write_nzp_file(
    const std::string& path,
    const FileHeader& header,
    const std::vector<uint8_t>& payload
);

/// Read header and payload from a file.
ErrorCode read_nzp_file(
    const std::string& path,
    FileHeader& outHeader,
    std::vector<uint8_t>& outPayload
);

} // namespace neurozip
