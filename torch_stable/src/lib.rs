//! Minimal hand written bindings for the Stable ABI functions.
//!
//! See <https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html>.
//!
//!
//! The file structure follows the upstream torch repository.
//! The root of this crate is roughly the `csrc` [directory](https://github.com/pytorch/pytorch/tree/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc)
//
// Docs aren't great, the the stable c++ api is well documented:
// https://github.com/pytorch/pytorch/tree/main/torch/csrc/stable
//
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h#L21

// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/macros.h#L46

// Hmm, the stable c++ interface actually goes through the call inteface; https://github.com/pytorch/pytorch/blob/e3d4dcc4adddff454ceeef71b3d735acc5e23344/torch/csrc/stable/ops.h#L625
//
//
// Bunch of tests here: https://github.com/pytorch/pytorch/tree/1c0fd99bc15e998eab3cee79588544a033f0e4df/test/cpp_extensions/libtorch_agn_2_10_extension/csrc
// empty is here: https://github.com/pytorch/pytorch/blob/1c0fd99bc15e998eab3cee79588544a033f0e4df/test/cpp_extensions/libtorch_agn_2_10_extension/csrc/my_empty.cpp#L10-L18
// Through; https://github.com/pytorch/pytorch/blob/1c0fd99bc15e998eab3cee79588544a033f0e4df/test/cpp_extensions/libtorch_agn_2_10_extension/csrc/my_empty.cpp#L10-L18

// ooh https://github.com/pytorch/extension-cpp/tree/1c325b202ae5e11de3cefb9a65be28f47949edd4
// they just pass a pytorch Tensor to a torch::stable::Tensor!? O_O

// Notes:
//   https://github.com/pytorch/pytorch/issues/174507
//   Lack of retrieving error messages, or even disabling the c++ backtrace.
//   new StableIValue is not portable, malloc != new? It's complex, valgrind lights up.
//   Why are sizes i64s? Same with enums being i32s?
//   Why doesn't ops have add? I tried a dispatch, but we can't because of the scalar value in the signature I think? See contrib.
//   Optional arguments for kernel launches are not supported it seems? https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/shim_common.cpp#L542-L545
pub mod aoti_torch;
pub mod contrib;
pub mod headeronly;
pub mod stable;
pub mod util;

pub use aoti_torch::{AOTI_TORCH_FAILURE, AOTI_TORCH_SUCCESS};
// Input param: AtenTensorHandle
// Output param: &mut AtenTensorHandle

pub use util::StableTorchResult;
// pub use util::{unsafe_call_bail, unsafe_call_dispatch_bail, unsafe_call_panic};

#[cfg(test)]
pub(crate) const RUN_SPAMMY_TESTS: bool = false;

/// This is the ABI version passed to the kernel dispatches.
pub const TORCH_ABI_VERSION: u64 = 0x20b000000000000; // as retrieved by _torch_abi_version, through test_aoti_torch_abi_version_print

include!(concat!(env!("OUT_DIR"), "/generated_consts.rs"));

/// Helper function that downtree users can call from their `build.rs` file:
///
/// ```rust,ignore
/// fn main() {
///    torch_stable::downtree_build_rs();
/// }
/// ```
///
/// This function prints the appropriate cargo flags to link the libraries / binaries of a downtree user with the
/// libraries.
/// It uses the feature flags `stable_torch` itself was built with, so be sure to propagate them all the way down.
pub fn downtree_build_rs() {
    println!("cargo:rustc-link-search=native={LIB_PATH}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", LIB_PATH);
    if FEATURE_CUDA {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-lib=torch_cuda");
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-ltorch");
    }
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-lib=torch_cpu");
}
