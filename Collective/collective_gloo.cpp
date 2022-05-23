#include <iostream>
#include <vector>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>

#include <c10d/ProcessGroupGloo.hpp>
torch::Tensor _emb_gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    int root){
        // std::stringstream ss;
        cout << "hello world!"<<endl;
    }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("emb_exchange", &_emb_gather, "Dyanmic GNN emb gather (gloo)");
}