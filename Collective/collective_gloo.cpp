#include <iostream>
#include <vector>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>

#include <c10d/GlooDeviceFactory.hpp>
#include "gloo.h"
#include <c10d/ProcessGroupGloo.hpp>

// uint32_t nextTag() {
//   return collectiveCounter_++;
// }

// std::shared_ptr<::gloo::Context> getContext(uint32_t tag) {
//   return contexts_[tag % contexts_.size()];
// }
// 
#define GENERATE_ALL_TYPES(type, func, args...)        \
  switch (type) {                                      \
    case ::at::ScalarType::Float:                      \
      func<float>(args);                               \
      break;                                           \
    case ::at::ScalarType::Double:                     \
      func<double>(args);                              \
      break;                                           \
    case ::at::ScalarType::Half:                       \
      func<gloo::float16>(args);                       \
      break;                                           \
    case ::at::ScalarType::Char:                       \
      func<int8_t>(args);                              \
      break;                                           \
    case ::at::ScalarType::Byte:                       \
      func<uint8_t>(args);                             \
      break;                                           \
    case ::at::ScalarType::Int:                        \
      func<int32_t>(args);                             \
      break;                                           \
    case ::at::ScalarType::Long:                       \
      func<int64_t>(args);                             \
      break;                                           \
    default:                                           \
      TORCH_CHECK(false, "Invalid scalar type"); \
  }

template <typename T, typename O>
void setInputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setInputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor) {
  opts.setInput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setOutputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor) {
  opts.setOutput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

at::Tensor pinnedLike(at::Tensor& tensor) {
  auto* allocator = at::detail::getCUDAHooks().getPinnedMemoryAllocator();
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
          tensor.sizes(), tensor.strides(), tensor.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  return at::empty({0}, tensor.options().device(at::kCPU))
      .set_(storage, 0, tensor.sizes(), tensor.strides());
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
        // Step 2: get the process rank and root rank
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("emb_exchange", &_emb_gather, "Dyanmic GNN emb gather (gloo)");
}