// Quick example to test if we can make a pytorch extension.
//
// There's a lot macro stuff here:
// https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L2

// So the entry appears to be that STABLE_TORCH_LIBRARY(extension_cpp, m) entry.
// https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L355
// Which seems to just instantiate this class; https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L130-L146
// WHich in turn calls the initFn on load. We an to the same.

#[used]
#[unsafe(link_section = ".init_array")]
static INIT_FUNC: extern "C" fn() = my_init_function;

extern "C" fn my_init_function() {
    // Your pre-main initialization logic here
    println!("test");
}
