#include "neurozip_c.h"

#include "../core/file_format.h"
#include "../core/model_interface.h"
#include "../models/tiny_lstm.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

struct nzp_model {
    std::unique_ptr<neurozip::TinyLstmModel> impl;
};

extern "C" {

nzp_model_t* nzp_model_load(const char* path)
{
    if (!path) return nullptr;
    auto modelPtr = std::make_unique<neurozip::TinyLstmModel>();
    if (!modelPtr->load_from_file(path)) {
        return nullptr;
    }
    auto wrapper = new nzp_model;
    wrapper->impl = std::move(modelPtr);
    return wrapper;
}

void nzp_model_free(nzp_model_t* model)
{
    delete model;
}

static nzp_error_t to_nzp_error(neurozip::ErrorCode e)
{
    using E = neurozip::ErrorCode;
    switch (e) {
        case E::Ok: return NZP_OK;
        case E::IoError: return NZP_ERR_IO;
        case E::InvalidFormat: return NZP_ERR_INVALID_FORMAT;
        case E::UnsupportedVersion: return NZP_ERR_UNSUPPORTED_VERSION;
        case E::ModelMismatch: return NZP_ERR_MODEL_MISMATCH;
        case E::CorruptData: return NZP_ERR_CORRUPT;
        default: return NZP_ERR_INTERNAL;
    }
}

static nzp_error_t compress_file_impl(
    const char* input_path,
    const char* output_path,
    const neurozip::ICompressionModel& model
) {
    // Read input
    std::ifstream ifs(input_path, std::ios::binary);
    if (!ifs) return NZP_ERR_IO;
    std::vector<uint8_t> data(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>()
    );

    neurozip::FileHeader header;
    header.originalSize = data.size();
    header.modelId = model.model_id();
    header.modelHash = model.model_hash();
    header.checksum = neurozip::crc32(data.data(), data.size());

    auto payload = neurozip::compress_buffer(model, data.data(), data.size());

    auto ec = neurozip::write_nzp_file(output_path, header, payload);
    return to_nzp_error(ec);
}

static nzp_error_t decompress_file_impl(
    const char* input_path,
    const char* output_path,
    const neurozip::ICompressionModel& model
) {
    neurozip::FileHeader header;
    std::vector<uint8_t> payload;
    auto ec = neurozip::read_nzp_file(input_path, header, payload);
    if (ec != neurozip::ErrorCode::Ok) {
        return to_nzp_error(ec);
    }

    // Check model id + hash match
    if (header.modelId != model.model_id() ||
        (header.modelHash != 0 && header.modelHash != model.model_hash())) {
        return NZP_ERR_MODEL_MISMATCH;
    }

    std::vector<uint8_t> out;
    if (!neurozip::decompress_buffer(model, payload.data(), payload.size(), header.originalSize, out)) {
        return NZP_ERR_CORRUPT;
    }

    // Verify checksum
    uint32_t crc = neurozip::crc32(out.data(), out.size());
    if (crc != header.checksum) {
        return NZP_ERR_CORRUPT;
    }

    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs) return NZP_ERR_IO;
    ofs.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!ofs) return NZP_ERR_IO;

    return NZP_OK;
}

nzp_error_t nzp_compress_file(
    const char* input_path,
    const char* output_path,
    const nzp_model_t* model
) {
    if (!input_path || !output_path || !model || !model->impl) {
        return NZP_ERR_INTERNAL;
    }
    return compress_file_impl(input_path, output_path, *model->impl);
}

nzp_error_t nzp_decompress_file(
    const char* input_path,
    const char* output_path,
    const nzp_model_t* model
) {
    if (!input_path || !output_path || !model || !model->impl) {
        return NZP_ERR_INTERNAL;
    }
    return decompress_file_impl(input_path, output_path, *model->impl);
}

const char* nzp_strerror(nzp_error_t err)
{
    switch (err) {
        case NZP_OK: return "ok";
        case NZP_ERR_IO: return "I/O error";
        case NZP_ERR_INVALID_FORMAT: return "invalid neurozip file format";
        case NZP_ERR_UNSUPPORTED_VERSION: return "unsupported neurozip format version";
        case NZP_ERR_MODEL_MISMATCH: return "model mismatch";
        case NZP_ERR_CORRUPT: return "corrupt compressed data";
        case NZP_ERR_INTERNAL: return "internal error";
        default: return "unknown error";
    }
}

} // extern "C"
