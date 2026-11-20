// Run the TorchScript Nicheformer embedder from C++.
//
// usage: nicheformer_embed <embedder.pt> <tokens.pt> <out_embeddings.pt> [batch_size]
//
//   embedder.pt  - saved from Python with torch.jit.script(NicheformerEmbedder(...)).save(...)
//   tokens.pt    - saved from Python with torch.save({'X': ..., 'modality': ..., 'assay': ..., 'specie': ...})
//   out.pt       - embeddings [n_cells, dim_model]; read back in Python with torch.load(...)

#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/serialize.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: nicheformer_embed <embedder.pt> <tokens.pt> <out.pt> [batch_size]\n";
        return 1;
    }
    const int64_t batch_size = argc == 5 ? std::stoll(argv[4]) : 4;  // small: attention is O(seq^2)
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "device: " << (device.is_cuda() ? "cuda" : "cpu") << "\n";

    try {
        // 1. Load the compiled model
        torch::jit::script::Module embedder = torch::jit::load(argv[1], device);
        embedder.eval();

        // 2. Load the token dict written by torch.save in Python
        c10::Dict<c10::IValue, c10::IValue> raw =
            torch::pickle_load(read_file(argv[2])).toGenericDict();
        c10::Dict<std::string, torch::Tensor> tokens;
        for (const auto& kv : raw) tokens.insert(kv.key().toStringRef(), kv.value().toTensor());
        const int64_t n_cells = tokens.at("X").size(0);
        std::cout << "cells: " << n_cells << ", tokens per cell: " << tokens.at("X").size(1) << "\n";

        // 3. Run in batches
        torch::InferenceMode guard;
        std::vector<torch::Tensor> outputs;
        for (int64_t start = 0; start < n_cells; start += batch_size) {
            const int64_t end = std::min(start + batch_size, n_cells);
            c10::Dict<std::string, torch::Tensor> batch;
            for (const auto& kv : tokens)
                batch.insert(kv.key(), kv.value().slice(0, start, end).to(device));
            outputs.push_back(embedder.forward({batch}).toTensor().to(torch::kCPU));
        }
        torch::Tensor emb = torch::cat(outputs, 0);
        std::cout << "embeddings: " << emb.sizes() << "\n";

        // 4. Save so Python can read it with torch.load
        std::vector<char> bytes = torch::pickle_save(emb);
        std::ofstream(argv[3], std::ios::binary).write(bytes.data(), bytes.size());
        std::cout << "saved " << argv[3] << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
