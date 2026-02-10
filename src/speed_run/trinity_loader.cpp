#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdint>

// Trinity Magic Header
const char MAGIC[] = "TRIN";

// De-quantization Look-Up Table (LUT)
// Mapping: 0->0, 1->1, 2->-1, 3->0 (unused)
const float LUT[4] = {0.0f, 1.0f, -1.0f, 0.0f};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./trinity_loader <path_to_model.bin>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << ">> Initiating Trinity Speed Run: " << model_path << std::endl;

    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        std::cerr << "!! Error: Could not open file." << std::endl;
        return 1;
    }

    // 1. Check Magic Header
    char header[4];
    file.read(header, 4);
    if (std::string(header, 4) != "TRIN") {
        std::cerr << "!! Error: Invalid Magic Header. Expected 'TRIN'." << std::endl;
        return 1;
    }
    std::cout << ">> Magic Header Verified: TRIN" << std::endl;

    // 2. Load & Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t total_weights = 0;
    size_t total_bytes = 0;
    int layer_count = 0;

    while (file.peek() != EOF) {
        // Read Block Size
        uint32_t block_size;
        file.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
        if (file.eof()) break;

        // Read Compressed Data
        std::vector<uint8_t> buffer(block_size);
        file.read(reinterpret_cast<char*>(buffer.data()), block_size);
        
        total_bytes += block_size;
        total_weights += block_size * 4; // 4 weights per byte
        layer_count++;

        // Quick Verification of first byte (De-quantization Check)
        if (layer_count == 1) {
            uint8_t b = buffer[0];
            std::cout << "   [Debug] Layer 1 First Byte: " << (int)b << " -> Weights: ";
            std::cout << LUT[(b >> 6) & 0x03] << ", ";
            std::cout << LUT[(b >> 4) & 0x03] << ", ";
            std::cout << LUT[(b >> 2) & 0x03] << ", ";
            std::cout << LUT[(b) & 0x03] << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << ">> SPEED RUN COMPLETE" << std::endl;
    std::cout << "   Layers Loaded: " << layer_count << std::endl;
    std::cout << "   Total Size:    " << total_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "   Total Params:  " << total_weights / 1000000.0 << " M" << std::endl;
    std::cout << "   Time Taken:    " << diff.count() << " s" << std::endl;
    std::cout << "   Bandwidth:     " << (total_bytes / 1024.0 / 1024.0) / diff.count() << " MB/s" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}
