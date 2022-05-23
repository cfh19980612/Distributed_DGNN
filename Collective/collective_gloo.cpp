#include <iostream>
#include <vector>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>

#include <c10d/ProcessGroupGloo.hpp>
#ifdef GLOO
#include <gloo.h>
#endif
namespace c10d{
// uint32_t nextTag() {
//   return collectiveCounter_++;
// }

// std::shared_ptr<::gloo::Context> getContext(uint32_t tag) {
//   return contexts_[tag % contexts_.size()];
// }

torch::Tensor _emb_gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const ::c10d::GatherOptions& opts){
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
        // Step 2: get other default information
        const auto root_rank = opts.rootRank
        cout<<root_rank<<endl;


        // const auto scalarType = inputs[0].scalar_type();
        // // 实例化GatherOptions类，传入参数为process的context
        // opts.setRoot(root);
        // opts.setTag(tag);

        // // Set single temporary tensor on root process.
        // // This is later scattered to the separate output tensors.
        // at::Tensor flatOutputTensor;
        // if (context->rank == root) {
        // flatOutputTensor = newLikeFlat(outputs[0]);
        // GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
        // }

        // // TODO: set single input tensor only on the required processes?
        // // Set single input tensor on all processes.
        // GENERATE_ALL_TYPES(scalarType, setInput, opts, inputs[0]);
        // gloo::gather(opts);

        // // Unflatten into output tensors on root process.
        // if (context->rank == root) {
        // for(const auto i : c10::irange(outputs[0].size())) {
        //     outputs[0][i].copy_(flatOutputTensor[i]);
        // }
        // }
    } // function

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("emb_exchange", &_emb_gather, "Dyanmic GNN emb gather (gloo)");
}
} // namespace