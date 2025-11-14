#include "neurozip_cpp.h"

namespace neurozip {

Model::~Model()
{
    if (model_) {
        nzp_model_free(model_);
        model_ = nullptr;
    }
}

bool Model::load(const std::string& path)
{
    if (model_) {
        nzp_model_free(model_);
        model_ = nullptr;
    }
    model_ = nzp_model_load(path.c_str());
    return model_ != nullptr;
}

nzp_error_t compress_file(
    const std::string& input_path,
    const std::string& output_path,
    const Model& model
) {
    if (!model.raw()) return NZP_ERR_INTERNAL;
    return nzp_compress_file(input_path.c_str(), output_path.c_str(), model.raw());
}

nzp_error_t decompress_file(
    const std::string& input_path,
    const std::string& output_path,
    const Model& model
) {
    if (!model.raw()) return NZP_ERR_INTERNAL;
    return nzp_decompress_file(input_path.c_str(), output_path.c_str(), model.raw());
}

} // namespace neurozip
