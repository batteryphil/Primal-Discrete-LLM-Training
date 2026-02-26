#include <torch/extension.h>

torch::Tensor tandem_forward_cuda(
    torch::Tensor base_idx,
    torch::Tensor fine_idx,
    torch::Tensor scale,
    torch::Tensor lut);

torch::Tensor tandem_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor base_idx,
    torch::Tensor fine_idx,
    torch::Tensor scale,
    torch::Tensor lut,
    torch::Tensor vote_buffer,
    float grad_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tandem_forward_cuda, "Primal Tandem forward (CUDA)");
    m.def("backward", &tandem_backward_cuda, "Primal Tandem backward (CUDA)");
}
