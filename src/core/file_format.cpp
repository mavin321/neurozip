#include "file_format.h"

#include <fstream>

namespace neurozip {

FileHeader::FileHeader()
    : magic(NZP_MAGIC),
      formatVersion(NZP_FORMAT_VERSION),
      modelId(0),
      flags(0),
      originalSize(0),
      checksum(0),
      modelHash(0),
      reserved(0) {}

static uint32_t crc32_table[256];
static bool crc32_initialized = false;

static void init_crc32()
{
    if (crc32_initialized) return;
    uint32_t poly = 0xEDB88320u;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int j = 0; j < 8; ++j) {
            if (c & 1)
                c = poly ^ (c >> 1);
            else
                c >>= 1;
        }
        crc32_table[i] = c;
    }
    crc32_initialized = true;
}

uint32_t crc32(const uint8_t* data, size_t len, uint32_t seed)
{
    init_crc32();
    uint32_t c = ~seed;
    for (size_t i = 0; i < len; ++i) {
        c = crc32_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return ~c;
}

ErrorCode write_nzp_file(
    const std::string& path,
    const FileHeader& header,
    const std::vector<uint8_t>& payload
) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return ErrorCode::IoError;
    }

    ofs.write(reinterpret_cast<const char*>(&header), sizeof(FileHeader));
    if (!ofs) return ErrorCode::IoError;

    if (!payload.empty()) {
        ofs.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
        if (!ofs) return ErrorCode::IoError;
    }

    return ErrorCode::Ok;
}

ErrorCode read_nzp_file(
    const std::string& path,
    FileHeader& outHeader,
    std::vector<uint8_t>& outPayload
) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return ErrorCode::IoError;
    }

    ifs.read(reinterpret_cast<char*>(&outHeader), sizeof(FileHeader));
    if (!ifs) return ErrorCode::IoError;

    if (outHeader.magic != NZP_MAGIC) {
        return ErrorCode::InvalidFormat;
    }
    if (outHeader.formatVersion != NZP_FORMAT_VERSION) {
        return ErrorCode::UnsupportedVersion;
    }

    // Read rest of file as payload
    ifs.seekg(0, std::ios::end);
    std::streampos endPos = ifs.tellg();
    std::streampos payloadStart = static_cast<std::streampos>(sizeof(FileHeader));
    if (endPos < payloadStart) {
        return ErrorCode::InvalidFormat;
    }
    size_t payloadSize = static_cast<size_t>(endPos - payloadStart);

    outPayload.resize(payloadSize);
    ifs.seekg(payloadStart, std::ios::beg);
    if (payloadSize > 0) {
        ifs.read(reinterpret_cast<char*>(outPayload.data()), static_cast<std::streamsize>(payloadSize));
        if (!ifs) return ErrorCode::IoError;
    }

    return ErrorCode::Ok;
}

} // namespace neurozip
