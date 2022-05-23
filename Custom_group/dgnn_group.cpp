#include "dgnn_group.hpp"

namespace c10d {


bool ProcessGroupDGNN::WorkDGNN::isCompleted() {
  return true;
}

bool ProcessGroupDGNN::WorkDGNN::isSuccess() const {
  return true;
}

bool ProcessGroupDGNN::WorkDGNN::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupDGNN::WorkDGNN::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
ProcessGroupDGNN::ProcessGroupDGNN(int rank, int size)
    : ProcessGroup(rank, size) {}

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  for (auto& outputTensorVec : outputTensors) {
      for (auto& outputTensor : outputTensorVec) {
          outputTensor.zero_();
      }
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkDGNN>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::_allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  for (auto& tensor : tensors) {
      tensor.zero_();
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDGNN>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::barrier(
    const BarrierOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  throw std::runtime_error("not supported");
}

// c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::gather(
//     std::vector<std::vector<at::Tensor>>& /* unused */,
//     std::vector<at::Tensor>& /* unused */,
//     const GatherOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }
/*
key part for customized gather function with objectives:
1. each group member can send different size of tensor for gather
2. not all members need to send tensor
3. cuda support
*/

namespace {

class AsyncGatherWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncGatherWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      int root,
      uint32_t tag)
      : ProcessGroupGloo::AsyncWork(outputs, "gloo:gather", inputs),
        context(context),
        outputs(outputs),
        inputs(inputs),
        root(root),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const int root;
  const uint32_t tag;

  void gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs) {
    const auto scalarType = inputs[0].scalar_type();
    // 实例化GatherOptions类，传入参数为process的context
    gloo::GatherOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set single temporary tensor on root process.
    // This is later scattered to the separate output tensors.
    at::Tensor flatOutputTensor;
    if (context->rank == root) {
      flatOutputTensor = newLikeFlat(outputs[0]);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    }

    // TODO: set single input tensor only on the required processes?
    // Set single input tensor on all processes.
    GENERATE_ALL_TYPES(scalarType, setInput, opts, inputs[0]);
    gloo::gather(opts);

    // Unflatten into output tensors on root process.
    if (context->rank == root) {
      for(const auto i : c10::irange(outputs[0].size())) {
        outputs[0][i].copy_(flatOutputTensor[i]);
      }
    }
  }

  void run() override {
    gather(outputs, inputs);
  }
};

// Note: current CUDA implementation holds the assumptions:
//     - inputs.size() is 1
//     - outputs.size() is 1
//     - the size of the nested output tensors is world size, i.e.,
//       outputs[0].size, is world size
class AsyncGatherCUDAWork : public AsyncGatherWork {
 public:
  AsyncGatherCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      int root,
      uint32_t tag)
      : AsyncGatherWork(context, outputs, inputs, root, tag) {
    initializeStreamsEvents(inputs, inputStreams, inputEvents);
    initializeStreamsEvents(outputs, outputStreams, outputEvents);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmpInputs.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for(const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(inputStreams[i]);
      tmpInputs.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }

    tmpOutputs.resize(outputs.size());
    for(const auto i : c10::irange(outputs.size())) {
      tmpOutputs[i].reserve(outputs[i].size());
      for(const auto j : c10::irange(outputs[i].size())) {
        tmpOutputs[i].push_back(pinnedLike(outputs[i][j]));
      }
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for(const auto i : c10::irange(inputs.size())) {
      inputStreams[i].synchronize();
    }

    for(const auto i : c10::irange(outputs.size())) {
      outputStreams[i].synchronize();
    }

    // Run gather on host side tensors.
    gather(tmpOutputs, tmpInputs);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for(const auto i : c10::irange(outputs.size())) {
      guard.reset_stream(outputStreams[i]);
      for(const auto j : c10::irange(outputs[i].size())) {
        outputs[i][j].copy_(tmpOutputs[i][j], /* non_blocking */ true);
      }
      outputEvents[i].record(outputStreams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for(const auto i : c10::irange(outputs.size())) {
      c10::Device device = outputs[i][0].device();
      outputEvents[i].block(c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmpInputs;
  std::vector<c10::Stream> inputStreams;
  std::vector<c10::Event> inputEvents;

  std::vector<std::vector<at::Tensor>> tmpOutputs;
  std::vector<c10::Stream> outputStreams;
  std::vector<c10::Event> outputEvents;
};

} // namespace

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupGloo::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertSingleElementInput(invalidArgument, inputs);
  assertDense(invalidArgument, inputs);

  if (getRank() == opts.rootRank) {
    if (outputs.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputs[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputs[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = inputs[0].options();
    const auto& sizes = inputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, outputs[0], options, sizes);
  } else {
    if (outputs.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
  }

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncGatherWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncGatherWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncGatherCUDAWork>(
        std::move(context), outputs, inputs, opts.rootRank, tag);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}
///

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("not supported");
}

// realize the group create funciton
c10::intrusive_ptr<ProcessGroup> ProcessGroupDGNN::createProcessGroupDGNN(
    const c10::intrusive_ptr<::c10d::Store>& /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<ProcessGroupDGNN>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // bind python and C++11, 'createProcessGroupDGNN' is the name of python API
  m.def("createProcessGroupDGNN", &ProcessGroupDGNN::createProcessGroupDGNN);
}

} // namespace c10d