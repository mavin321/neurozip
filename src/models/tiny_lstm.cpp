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
// uint32 reserved
// Then float32 weights in order:
// w_ih (4H*I), w_hh (4H*H),
// b_ih (4H), b_hh (4H),
// w_out (256*H), b_out (256)
bool TinyLstmModel::load_from_file(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    uint32_t inputSize = 0, hiddenSize = 0, numLayers = 0, reserved = 0;
    ifs.read((char*)&inputSize, sizeof(uint32_t));
    ifs.read((char*)&hiddenSize, sizeof(uint32_t));
    ifs.read((char*)&numLayers, sizeof(uint32_t));
    ifs.read((char*)&reserved, sizeof(uint32_t));

    if (!ifs) return false;
    if (inputSize != 256 || numLayers != 1) return false;

    weights_.inputSize = inputSize;
    weights_.hiddenSize = hiddenSize;
    weights_.numLayers = numLayers;

    size_t H = hiddenSize;
    size_t I = inputSize;

    auto read_vec = [&](std::vector<float>& v, size_t n) {
        v.resize(n);
        ifs.read((char*)v.data(), (std::streamsize)(n * sizeof(float)));
        return (bool)ifs;
    };

    if (!read_vec(weights_.w_ih, 4 * H * I)) return false;
    if (!read_vec(weights_.w_hh, 4 * H * H)) return false;
    if (!read_vec(weights_.b_ih, 4 * H)) return false;
    if (!read_vec(weights_.b_hh, 4 * H)) return false;
    if (!read_vec(weights_.w_out, 256 * H)) return false;
    if (!read_vec(weights_.b_out, 256)) return false;

    // Hash all weights (FNV-1a)
    uint64_t hash = 1469598103934665603ull;
    auto hash_floats = [&](const std::vector<float>& v) {
        const uint8_t* bytes = (const uint8_t*)v.data();
        size_t n = v.size() * sizeof(float);
        for (size_t i = 0; i < n; i++) {
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
    modelId_ = 1;

    return true;
}

std::unique_ptr<ModelContext> TinyLstmModel::create_context() const
{
    auto ctx = std::make_unique<ModelContext>();

    // Initialize arrays to 0
    for (size_t i = 0; i < 256; i++) {
        ctx->h[i] = 0.0f;
        ctx->c[i] = 0.0f;
    }

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

    if (outHidden.size() != H)
        outHidden.assign(H, 0.0f);

    const float* hPrev = ctx.h;
    float* cPrev = ctx.c;

    const auto& w_ih = weights_.w_ih;
    const auto& w_hh = weights_.w_hh;
    const auto& b_ih = weights_.b_ih;
    const auto& b_hh = weights_.b_hh;

    std::vector<float> gates(4 * H);

    // gates = bi + bh
    for (size_t i = 0; i < 4 * H; i++)
        gates[i] = b_ih[i] + b_hh[i];

    // W_ih * x  (one-hot input)
    for (size_t row = 0; row < 4 * H; row++) {
        gates[row] += w_ih[row * I + xByte];
    }

    // W_hh * hPrev
    for (size_t row = 0; row < 4 * H; row++) {
        float acc = 0.0f;
        const float* Wrow = &w_hh[row * H];
        for (size_t j = 0; j < H; j++)
            acc += Wrow[j] * hPrev[j];
        gates[row] += acc;
    }

    // Split gates
    float* iGate = gates.data();
    float* fGate = gates.data() + H;
    float* gGate = gates.data() + 2 * H;
    float* oGate = gates.data() + 3 * H;

    // LSTM update
    for (size_t i = 0; i < H; i++) {
        float i_t = sigmoid(iGate[i]);
        float f_t = sigmoid(fGate[i]);
        float g_t = std::tanh(gGate[i]);
        float o_t = sigmoid(oGate[i]);

        cPrev[i] = f_t * cPrev[i] + i_t * g_t;
        outHidden[i] = o_t * std::tanh(cPrev[i]);
    }

    // Copy back hidden state to ModelContext
    for (size_t i = 0; i < H; i++) {
        ctx.h[i] = outHidden[i];
    }
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

    float logits[256];

    // logits = W_out*h + b
    for (size_t i = 0; i < 256; i++) {
        float acc = b_out[i];
        const float* row = &w_out[i * H];
        for (size_t j = 0; j < H; j++)
            acc += row[j] * hNew[j];
        logits[i] = acc;
    }

    // Softmax
    float maxLogit = logits[0];
    for (size_t i = 1; i < 256; i++)
        if (logits[i] > maxLogit) maxLogit = logits[i];

    float sum = 0.0f;
    for (size_t i = 0; i < 256; i++) {
        float e = std::exp(logits[i] - maxLogit);
        outProbs[i] = e;
        sum += e;
    }

    if (sum <= 0.0f) {
        float p = 1.0f / 256.0f;
        for (size_t i = 0; i < 256; i++)
            outProbs[i] = p;
        return;
    }

    float invSum = 1.0f / sum;
    for (size_t i = 0; i < 256; i++)
        outProbs[i] *= invSum;
}

} // namespace neurozip
