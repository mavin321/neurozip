#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nzp_model nzp_model_t;

typedef enum {
    NZP_OK = 0,
    NZP_ERR_IO,
    NZP_ERR_INVALID_FORMAT,
    NZP_ERR_UNSUPPORTED_VERSION,
    NZP_ERR_MODEL_MISMATCH,
    NZP_ERR_CORRUPT,
    NZP_ERR_INTERNAL
} nzp_error_t;

/// Load a Tiny LSTM model from a binary file.
nzp_model_t* nzp_model_load(const char* path);

/// Free a model object.
void nzp_model_free(nzp_model_t* model);

/// Compress a file (input_path) into output_path.
/// Returns NZP_OK on success.
nzp_error_t nzp_compress_file(
    const char* input_path,
    const char* output_path,
    const nzp_model_t* model
);

/// Decompress a .nzp file into output_path.
/// Returns NZP_OK on success.
nzp_error_t nzp_decompress_file(
    const char* input_path,
    const char* output_path,
    const nzp_model_t* model
);

/// Get human-readable error string.
const char* nzp_strerror(nzp_error_t err);

#ifdef __cplusplus
}
#endif
