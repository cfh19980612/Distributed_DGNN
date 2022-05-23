#include <iostream>
#include <vector>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>

#include <c10d/ProcessGroupGloo.hpp>
namespace {
torch::Tensor _emb_gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    int root){
        // std::stringstream ss;
        throw std::runtime_error("No implementation!");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("emb_exchange", &_emb_gather, "Dyanmic GNN emb gather (gloo)");
}