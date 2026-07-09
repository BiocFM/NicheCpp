#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    double value = static_cast<double>(bytes);
    int unit = 0;

    while (value >= 1024.0 && unit < 3) {
        value /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << " " << units[unit];
    return oss.str();
}

void print_tensor_info(const std::string& name, const torch::Tensor& t) {
    std::cout
        << "  " << name
        << " | shape=" << t.sizes()
        << " | dtype=" << t.scalar_type()
        << " | numel=" << t.numel()
        << " | bytes=" << format_bytes(t.numel() * t.element_size())
        << std::endl;
}

void inspect_model(torch::jit::script::Module& module) {
    int64_t total_param_tensors = 0;
    int64_t total_param_values = 0;
    size_t total_param_bytes = 0;

    int64_t total_buffer_tensors = 0;
    int64_t total_buffer_values = 0;
    size_t total_buffer_bytes = 0;

    std::cout << "\n=== Parameters ===" << std::endl;

    for (const auto& p : module.named_parameters(/*recurse=*/true)) {
        const std::string& name = p.name;
        const torch::Tensor& t = p.value;

        print_tensor_info(name, t);

        total_param_tensors++;
        total_param_values += t.numel();
        total_param_bytes += t.numel() * t.element_size();
    }

    std::cout << "\n=== Buffers ===" << std::endl;

    for (const auto& b : module.named_buffers(/*recurse=*/true)) {
        const std::string& name = b.name;
        const torch::Tensor& t = b.value;

        print_tensor_info(name, t);

        total_buffer_tensors++;
        total_buffer_values += t.numel();
        total_buffer_bytes += t.numel() * t.element_size();
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "parameter tensors: " << total_param_tensors << std::endl;
    std::cout << "parameter values : " << total_param_values << std::endl;
    std::cout << "parameter memory : " << format_bytes(total_param_bytes) << std::endl;

    std::cout << "buffer tensors   : " << total_buffer_tensors << std::endl;
    std::cout << "buffer values    : " << total_buffer_values << std::endl;
    std::cout << "buffer memory    : " << format_bytes(total_buffer_bytes) << std::endl;

    std::cout << "total values     : " << total_param_values + total_buffer_values << std::endl;
    std::cout << "total memory     : " << format_bytes(total_param_bytes + total_buffer_bytes) << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "Loading model..." << std::endl;
        torch::jit::script::Module module = torch::jit::load("nicheformer_model/nicheformer_embeddings.pt");
        module.eval();

        inspect_model(module);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}