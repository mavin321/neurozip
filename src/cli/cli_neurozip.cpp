#include <iostream>
#include <string>
#include "../api/neurozip_cpp.h"

static void print_usage() {
    std::cout << "Usage: neurozip [options] <input-file>\n"
              << "Options:\n"
              << "  -o <file>       Output file (.nzp)\n"
              << "  -m <model.bin>  Tiny LSTM model file\n"
              << "  -v              Verbose output\n";
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string inputPath;
    std::string outputPath;
    std::string modelPath;
    bool verbose = false;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (a == "-m" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (a == "-v") {
            verbose = true;
        } else if (a[0] == '-') {
            print_usage();
            return 1;
        } else {
            inputPath = a;
        }
    }

    if (inputPath.empty()) {
        print_usage();
        return 1;
    }

    if (outputPath.empty()) {
        outputPath = inputPath + ".nzp";
    }

    if (modelPath.empty()) {
        std::cerr << "Error: You must specify a model file with -m\n";
        return 1;
    }

    if (verbose) {
        std::cout << "Loading model: " << modelPath << "\n";
    }

    neurozip::Model model(modelPath);
    if (!model.valid()) {
        std::cerr << "Failed to load model: " << modelPath << "\n";
        return 1;
    }

    if (verbose) {
        std::cout << "Compressing " << inputPath << " -> " << outputPath << "\n";
    }

    auto err = neurozip::compress_file(inputPath, outputPath, model);

    if (err != NZP_OK) {
        std::cerr << "Compression error: " << nzp_strerror(err) << "\n";
        return 1;
    }

    if (verbose) {
        std::cout << "OK\n";
    }

    return 0;
}
