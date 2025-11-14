#include <iostream>
#include <string>
#include <vector>

#include "../core/file_format.h"

static void usage() {
    std::cout << "Usage: neurozip-inspect <file.nzp>\n";
}

static void print_header(const neurozip::FileHeader& h) {
    std::cout << "Magic:          0x" << std::hex << h.magic << std::dec << "\n";
    std::cout << "Format version: " << (int)h.formatVersion << "\n";
    std::cout << "Model ID:       " << h.modelId << "\n";
    std::cout << "Model Hash:     " << h.modelHash << "\n";
    std::cout << "Original size:  " << h.originalSize << "\n";
    std::cout << "CRC32:          0x" << std::hex << h.checksum << std::dec << "\n";
    std::cout << "Reserved:       " << h.reserved << "\n";
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        usage();
        return 1;
    }

    const std::string path = argv[1];

    neurozip::FileHeader h;
    std::vector<uint8_t> payload;

    auto err = neurozip::read_nzp_file(path, h, payload);
    if (err != neurozip::ErrorCode::Ok) {
        std::cerr << "Error: cannot read file: " << (int)err << "\n";
        return 1;
    }

    print_header(h);

    std::cout << "Payload bytes:  " << payload.size() << "\n";

    return 0;
}
