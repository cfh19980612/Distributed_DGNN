#include <iostream>
#include <vector>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>

#include <nccl.h>

namespace {
#include <c10d/ProcessGroupGloo.hpp>

uint32_t nextTag() {
  return collectiveCounter_++;
}

std::shared_ptr<::gloo::Context> getContext(uint32_t tag) {
  return contexts_[tag % contexts_.size()];
}

torch::Tensor _emb_gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    std::vector<std::vector<int>>& target_members,
    const GatherOptions& opts) {
        // throw std::runtime_error("No implementation!");
        static auto invalidArgument = [](const std::string& msg) {
            TORCH_CHECK(false, "ProcessGroupGloo::broadcast: " + msg);
        };
        // Step 1: Check the environment
        const auto& device = inputs[0].device();
        switch (device.type()) {
            case at::kCPU:
                break;
            case at::kCUDA:
            // If the user gave us a CUDA tensor then CUDA must be loaded.
                TORCH_INTERNAL_ASSERT(at::hasCUDA());
                break;
            default:
                invalidArgument(str("unsupported device type ", device.type()));
        }
        // Step 2: get the process rank
        const int rank = getRank()
        const int root = opts.rootRank
        
        // Step 3: create receive tensor list
        const auto scalarType = inputs[0].scalar_type();
        at::Tensor flatOutputTensor;
        if (rank == root) {
            flatOutputTensor = newLikeFlat(outputs[0]);  // create a tensor list to receive tensors
            GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
        }
        // 'GENERATE_ALL_TYPES' see https://github.com/pytorch/pytorch/blob/master/torch/csrc/distributed/c10d/ProcessGroupGloo.cpp

        // TODO: set single input tensor only on the required processes?
        // Step 4: Set single input tensor on all processes.
        if (rank in target_members) {
            GENERATE_ALL_TYPES(scalarType, setInput, opts, inputs[0]);
            gloo::gather(opts);
        }
        // Unflatten into output tensors on root process.
        if (rank == root) {
            for(const auto i : c10::irange(outputs[0].size())) {
                outputs[0][i].copy_(flatOutputTensor[i]);
            }
        }
    } // function
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("emb_exchange", &_emb_gather, "Dyanmic GNN emb gather (gloo)");
}
} // namespace