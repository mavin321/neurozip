#pragma once

#include <memory>
#include <string>

#include "neurozip_c.h"

namespace neurozip {

class Model {
public:
    Model() = default;
    explicit Model(const std::string& path) { load(path); }
    ~Model();

    bool load(const std::string& path);
    bool valid() const { return model_ != nullptr; }

    nzp_model_t* raw() const { return model_; }

private:
    nzp_model_t* model_ = nullptr;
};

nzp_error_t compress_file(
    const std::string& input_path,
    const std::string& output_path,
    const Model& model
);

nzp_error_t decompress_file(
    const std::string& input_path,
    const std::string& output_path,
    const Model& model
);

} // namespace neurozip
