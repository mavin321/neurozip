#include "tiny_lstm.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>

namespace neurozip {

TinyLstmModel::TinyLstmModel()
    : modelId_(1),
      modelHash_(0)
{
}

// Simple binary format:
// uint32 inputSize
// uint32 hiddenSize
// uint32 numLayers (must be 1)
// uint32 reserved (unused)
// then weights in order:
// w_ih (4H*I), w_hh (4H*H), b_ih (4H), b_hh (4H), w_out (256*H), b_out (256)
// all as float32 (little-endian)
bool TinyLstmModel::load_from_file(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    uint32_t inputSize = 0, hiddenSize = 0, numLayers = 0, reserved = 0;
    ifs.read(reinterpret_cast<char*>(&inputSize), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(&hiddenSize), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(&numLayers), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(&reserved), sizeof(uint32_t));
    if (!ifs) return false;

    if (inputSize != 256 || numLayers != 1) {
        return false;
    }

    weights_.inputSize = inputSize;
    weights_.hiddenSize = hiddenSize;
    weights_.numLayers = numLayers;

    size_t H = hiddenSize;
    size_t I = inputSize;

    size_t sz_w_ih = 4 * H * I;
    size_t sz_w_hh = 4 * H * H;
    size_t sz_b_ih = 4 * H;
    size_t sz_b_hh = 4 * H;
    size_t sz_w_out = 256 * H;
    size_t sz_b_out = 256;

    auto read_vec = [&](std::vector<float>& v, size_t n) {
        v.resize(n);
        ifs.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
        return static_cast<bool>(ifs);
    };

    if (!read_vec(weights_.w_ih, sz_w_ih)) return false;
    if (!read_vec(weights_.w_hh, sz_w_hh)) return false;
    if (!read_vec(weights_.b_ih, sz_b_ih)) return false;
    if (!read_vec(weights_.b_hh, sz_b_hh)) return false;
    if (!read_vec(weights_.w_out, sz_w_out)) return false;
    if (!read_vec(weights_.b_out, sz_b_out)) return false;

    // Compute a simple hash over all weights (not cryptographic).
    uint64_t hash = 1469598103934665603ull; // FNV-1a offset
    auto hash_floats = [&](const std::vector<float>& v) {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(v.data());
        size_t nbytes = v.size() * sizeof(float);
        for (size_t i = 0; i < nbytes; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };

    hash_floats(weights_.w_ih);
    hash_floats(weights_.w_hh);
    hash_floats(weights_.b_ih);
    hash_floats(weights_.b_hh);
    hash_floats(weights_.w_out);
    hash_floats(weights_.b_out);

    modelHash_ = hash;
    // modelId_ could be encoded in file if you want; for now just 1.
    modelId_ = 1;

    return true;
}

std::unique_ptr<ModelContext> TinyLstmModel::create_context() const
{
    auto ctx = std::make_unique<ModelContext>();
    ctx->h.assign(weights_.hiddenSize, 0.0f);
    ctx->c.assign(weights_.hiddenSize, 0.0f);
    return ctx;
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

void TinyLstmModel::step(
    ModelContext& ctx,
    uint8_t xByte,
    std::vector<float>& outHidden
) const
{
    size_t H = weights_.hiddenSize;
    size_t I = weights_.inputSize;

    // One-hot input vector x: length I, 1 at xByte.
    // We compute W_ih * x by selecting the column corresponding to xByte.

    const auto& w_ih = weights_.w_ih;
    const auto& w_hh = weights_.w_hh;
    const auto& b_ih = weights_.b_ih;
    const auto& b_hh = weights_.b_hh;

    const std::vector<float>& hPrev = ctx.h;
    std::vector<float>& cPrev = ctx.c;

    if (outHidden.size() != H) outHidden.assign(H, 0.0f);

    std::vector<float> gates(4 * H);

    // gates = b_ih + b_hh
    for (size_t i = 0; i < 4 * H; ++i) {
        gates[i] = b_ih[i] + b_hh[i];
    }

    // Add W_ih * x
    // Column xByte of W_ih (4H x I)
    for (size_t row = 0; row < 4 * H; ++row) {
        gates[row] += w_ih[row * I + static_cast<size_t>(xByte)];
    }

    // Add W_hh * hPrev
    for (size_t row = 0; row < 4 * H; ++row) {
        float acc = 0.0f;
        const float* wrow = &w_hh[row * H];
        for (size_t j = 0; j < H; ++j) {
            acc += wrow[j] * hPrev[j];
        }
        gates[row] += acc;
    }

    // Split gates: i, f, g, o
    float* iGate = gates.data();
    float* fGate = gates.data() + H;
    float* gGate = gates.data() + 2 * H;
    float* oGate = gates.data() + 3 * H;

    for (size_t i = 0; i < H; ++i) {
        float i_t = sigmoid(iGate[i]);
        float f_t = sigmoid(fGate[i]);
        float g_t = std::tanh(gGate[i]);
        float o_t = sigmoid(oGate[i]);

        cPrev[i] = f_t * cPrev[i] + i_t * g_t;
        outHidden[i] = o_t * std::tanh(cPrev[i]);
    }

    ctx.h = outHidden;
}

void TinyLstmModel::predict_next(
    ModelContext& ctx,
    uint8_t prevByte,
    float* outProbs,
    size_t outSize
) const
{
    if (outSize < 256) return;

    size_t H = weights_.hiddenSize;
    std::vector<float> hNew(H);

    step(ctx, prevByte, hNew);

    const auto& w_out = weights_.w_out;
    const auto& b_out = weights_.b_out;

    // logits = W_out * h + b_out
    float logits[256];
    for (size_t i = 0; i < 256; ++i) {
        float acc = b_out[i];
        const float* row = &w_out[i * H];
        for (size_t j = 0; j < H; ++j) {
            acc += row[j] * hNew[j];
        }
        logits[i] = acc;
    }

    // Softmax
    float maxLogit = logits[0];
    for (size_t i = 1; i < 256; ++i) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < 256; ++i) {
        float e = std::exp(logits[i] - maxLogit);
        outProbs[i] = e;
        sum += e;
    }
    if (sum <= 0.0f) {
        // fallback uniform
        float p = 1.0f / 256.0f;
        for (size_t i = 0; i < 256; ++i) outProbs[i] = p;
    } else {
        float invSum = 1.0f / sum;
        for (size_t i = 0; i < 256; ++i) outProbs[i] *= invSum;
    }
}

} // namespace neurozip
