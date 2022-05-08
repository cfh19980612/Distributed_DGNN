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

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDGNN::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

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
  // bind python and C++11, 'createProcessGroupDGNN' is the naem of python API
  m.def("createProcessGroupDGNN", &ProcessGroupDGNN::createProcessGroupDGNN);
}

} // namespace c10d