#pragma once

#include "core/model_interface.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace neurozip {

struct LstmWeights {
    uint32_t inputSize;   // should be 256
    uint32_t hiddenSize;
    uint32_t numLayers;

    // For simplicity, we assume a single-layer LSTM:
    // W_ih: [4*H, I]
    // W_hh: [4*H, H]
    // b_ih: [4*H]
    // b_hh: [4*H]
    std::vector<float> w_ih;
    std::vector<float> w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;

    // Output layer: logits = W_out * h + b_out
    // W_out: [256, H]
    // b_out: [256]
    std::vector<float> w_out;
    std::vector<float> b_out;
};



class TinyLstmModel : public ICompressionModel {
public:
    TinyLstmModel();
    ~TinyLstmModel() override = default;

    bool load_from_file(const std::string& path);

    std::unique_ptr<ModelContext> create_context() const override;

    void predict_next(
        ModelContext& ctx,
        uint8_t prevByte,
        float* outProbs,
        size_t outSize
    ) const override;

    uint32_t model_id() const override { return modelId_; }
    uint64_t model_hash() const override { return modelHash_; }

private:
    LstmWeights weights_;
    uint32_t modelId_;
    uint64_t modelHash_;

    void step(
        ModelContext& ctx,
        uint8_t xByte,
        std::vector<float>& outHidden
    ) const;
};

} // namespace neurozip
