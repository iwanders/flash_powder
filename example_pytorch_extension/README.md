# Pure Rust PyTorch extension

Quick test of how to write a PyTorch extension using the stable ABI.

Some docs on argument handling here: [the impl documentation](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L63-L84) we need to own the memory and free it.

This crate is built as an `crate-type=["cdylib"]` type crate such that it creates a `.so` file that we can then load from PyTorch.

It's full of unsafe, this is really just a test to see whether this was feasible and I have no use care for it.

See [./load.py](load.py) for how this is used from the Python side.

## Notes on C++

The following simple example has quite a bit of macro's going on, they're [here](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L2). 

This example:

```cpp
torch::stable::Tensor mymuladd_cpu(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    double c) {
  return torch::stable::matmul(a, b);
}

STABLE_TORCH_LIBRARY(extension_cpp, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}


STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("mymuladd", TORCH_BOX(&mymuladd_cpu));
}
```

Expands into (`gcc -E`):

```cpp
torch::stable::Tensor mymuladd_cpu(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    double c) {
# 31 "<snip>/muladd.cpp"
  return torch::stable::matmul(a, b);
}

static void STABLE_TORCH_LIBRARY_init_extension_cpp( torch::stable::detail::StableLibrary&); static const torch::stable::detail::StableTorchLibraryInit STABLE_TORCH_LIBRARY_static_init_extension_cpp( torch::stable::detail::StableLibrary::Kind::DEF, &STABLE_TORCH_LIBRARY_init_extension_cpp, "extension_cpp", nullptr, "<snip>/muladd.cpp", 34); void STABLE_TORCH_LIBRARY_init_extension_cpp(torch::stable::detail::StableLibrary& m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}


static void STABLE_TORCH_LIBRARY_IMPL_init_extension_cpp_CPU_0(torch::stable::detail::StableLibrary&); static const torch::stable::detail::StableTorchLibraryInit STABLE_TORCH_LIBRARY_IMPL_static_init_extension_cpp_CPU_0( torch::stable::detail::StableLibrary::Kind::IMPL, &STABLE_TORCH_LIBRARY_IMPL_init_extension_cpp_CPU_0, "extension_cpp", "CPU", "<snip>/muladd.cpp", 41); void STABLE_TORCH_LIBRARY_IMPL_init_extension_cpp_CPU_0( torch::stable::detail::StableLibrary & m) {
  m.impl("mymuladd", torch::stable::detail::boxer< std::remove_pointer_t<std::remove_reference_t<decltype(&mymuladd_cpu)>>, (&mymuladd_cpu)>::boxed_fn);
}
```

Notice there's two static objects here that are merely used to trigger the registration.

The
```cpp
torch::stable::detail::boxer< std::remove_pointer_t<std::remove_reference_t<decltype(&mymuladd_cpu)>>, (&mymuladd_cpu)>::boxed_fn
```

Is just metaprogramming to handle the arguments nicely, ultimately it just produces a function [of signature](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L227-L230):

```cpp
void boxed_fn(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs)
```
