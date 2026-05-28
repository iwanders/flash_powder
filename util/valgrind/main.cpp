

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>

#include <optional>
#include <iostream>

using torch::stable::Tensor;


void run_torch_stable_empty() {
  std::cout << "suppression@torch::stable::empty" << std::endl;
  torch::stable::empty(torch::headeronly::HeaderOnlyArrayRef<int64_t>{1,1,4,4}, {}, {}, {}, {}, {});
}
void run_torch_stable_to() {
  std::cout << "suppression@torch::stable::to" << std::endl;
  const auto t = torch::stable::empty(torch::headeronly::HeaderOnlyArrayRef<int64_t>{1,1,4,4}, {}, {}, {}, {}, {});
  // https://github.com/pytorch/pytorch/blob/2deace1f225e3ff25d776971def33ca59209c60a/torch/csrc/stable/ops.h#L824
  torch::stable::to(t, {torch::headeronly::ScalarType::Float} );
}

void run_torch_stable_empty_cuda() {
  std::cout << "suppression@run_torch_stable_empty" << std::endl;
  const std::optional<torch::stable::Device> device = torch::stable::Device("cuda");
  torch::stable::empty(torch::headeronly::HeaderOnlyArrayRef<int64_t>{1,1,4,4}, {}, {}, device, {}, {});
}

void run_torch_stable_fill(){
  std::cout << "suppression@dispatch:aten::fill_Tensor" << std::endl;

  const auto self = torch::stable::empty(torch::headeronly::HeaderOnlyArrayRef<int64_t>{5,5}, {}, {}, {}, {}, {});
  // let mut t = Tensor::zeros(&[5, 5], &Default::default())?;

  AtenTensorHandle ret0;
  STABLE_TORCH_ERROR_CODE_CHECK( aoti_torch_scalar_to_tensor_float32(3.3, &ret0));
  const auto v =  torch::stable::Tensor(ret0);


  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(v)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::fill", "Tensor", stack.data(), TORCH_ABI_VERSION));
  (void)torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

int main(int argc, char* argv[]){
  run_torch_stable_empty();
  run_torch_stable_to();
  run_torch_stable_fill();
  return 0;
}
