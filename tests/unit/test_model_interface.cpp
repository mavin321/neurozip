#include <cassert>
#include <iostream>
#include <cstring>
#include "../../src/core/model_interface.h"

using namespace neurozip;

// Fake model that always predicts uniform distribution
class DummyModel : public ICompressionModel {
public:
    std::unique_ptr<ModelContext> create_context() const override {
        return std::make_unique<ModelContext>();
    }
    void predict_next(ModelContext&, uint8_t, float* out, size_t) const override {
        for (int i = 0; i < 256; i++) out[i] = 1.0f / 256.0f;
    }
    uint32_t model_id() const override { return 99; }
    uint64_t model_hash() const override { return 0; }
};

int main() {
    std::cout << "[test_model_interface] Running...\n";

    DummyModel model;

    const char* text = "hello world";
    size_t n = strlen(text);

    auto comp = compress_buffer(model, (const uint8_t*)text, n);

    std::vector<uint8_t> out;
    bool ok = decompress_buffer(model, comp.data(), comp.size(), n, out);

    assert(ok == true);
    assert(std::string(out.begin(), out.end()) == text);

    std::cout << "[test_model_interface] OK\n";
    return 0;
}
