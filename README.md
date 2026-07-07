## Flash Powder

> What makes light and works through oxidization? Flash Powder.


A very minimal rust wrapper for libtorch, using the [Torch Stable API](https://docs.pytorch.org/cppdocs/stable.html) only.
This is mostly my project to gain a better understanding of how (lib/py)torch works under the hood. I do not recommend using this.

The stable ABI doesn't expose all functionality of libtorch, but a surprising amount of functionality is available,
especially if the goal is just to do inference. The [example_vgg](./example_vgg) crate holds an implementation of vgg11.

This was developed for doing inference with an U-net in my [overlay_segmenter](https://github.com/iwanders/overlay_segmenter/).

### Approach

It follows the rust semantics as closely as possible. This means;

- No unsafe in the public interface, safe behaviour as you'd expect.
- No interior mutability, all methods are const correct.
- Modifying one tensor will not modify another, unless it has a mutable borrow.
- Rust style lifetimes on tensors, either tied together with an explicit lifetime, or completely separate.

There are three structures fundamental to achieving this:

- `Tensor`; Owning tensor, this owns the data. (think `Vec<u8>`)
- `Ten<'_>`; Const borrow of Tensor, this has a parent, its lifetime cannot exceed the parent. (think `&[u8]`)
- `TenMut<'_>`; Mutable borrow of Tensor, this has a mutable parent, its lifetime cannot exceed the parent. (think `&mut [u8]`)

Under the hood, each of these is a `StableTensor` and its own tensor handle on the LibTorch side.

This doesn't map perfectly to Torch's operations, for example the `.to()` method in libtorch sometimes returns a copy, but not always.
So there's some arbitrary choices here, like `.to()` in this crate  always makes a copy.

### flash_powder

The main high-level and safe interface lives in the [flash_powder](./flash_powder) crate.

The crate is fairly well documented, here's an overview of existing functionality to get an idea of the semantics as well as the location in crate, most examples are copied from the unit tests.
All functions or methods that can fail return a `Result`, which when it fails holds an `anyhow::Error` with the message that was returned by the stable API.

Creating a tensor can be done with the [conversion](flash_powder/src/conversion.rs) module through `TryInto<Tensor>`:
```rust
// Convert a scalar like so:
let d: Tensor = 5i64.try_into()?;
assert_eq!(d.dim(), 0);
assert_eq!(d.i64_ref(&[])?, &5);
// Or create a 2D Tensor with some floats;
let d: Tensor = [[5.0f32, 3.0, 5.0], [1.0, 2.0, 0.0]].try_into()?;
assert_eq!(d.sizes(), &[2, 3]);

let d_as_u8 = d.to(&fp::DType::U8.into())?;
let u8_on_gpu = as_u8.to(&fp::Device::CUDA.into())?;
```

Or any of the [factory](flash_powder/src/factory.rs) trait methods:
```rust
let a = Tensor::empty(&[5, 5], &Default::default()); // Defaults to cpu, f32
let t = Tensor::randn(&[3, 3], &fp::Device::CPU.into())?; // We can give it a device to create it on
let e = Tensor::zeros(&[6, 6], &fp::DType::U8.into())?; // Or specify a type (or mix these options).
```

The properties of a tensor, like `dtype()`, `device()` and `sizes()` are all provided by the `TensorProperties` trait from the [properties](flash_powder/src/properties.rs) module.

Data access to the data contained in the Tensor is provided through the [data](flash_powder/src/data.rs) module, through the `DataRef` and `DataMut` traits.
This exposes (typed) slices created from the tensor's data using [zerocopy](https://docs.rs/zerocopy/latest/zerocopy/).
- `as_<T>()`: Access to the value stored in a scalar tensor:  `&T`
- `<T>_ref(indices: &[usize])`: Index into the storage to return a reference to a value at the provided index position: `&T`
- `<T>s_ref()`: Access to the entire slice of values: `&[T]`
- `as_<T>_mut()`, `<T>_mut(indices: &[usize])`, `<T>s_mut()`: Mutable flavours of these.


The `Tensor` and `Ten` implement the `CoreMethods` trait from the [core_methods](flash_powder/src/core_methods.rs) module.
This provides functionality like `flatten`, `mul`, `permute` and other operations.

```rust
let t = Tensor::from(&[0.2015f32, -0.4255, 2.6087])?;
let factor: Tensor = 100.0.try_into()?;
let r = t.mul(&factor)?;
assert_eq!(r.sizes(), &[3]);
assert_eq!(
    r.f32s_ref()?,
    &[20.149999618530273f32, -42.54999923706055, 260.8699951171875]
);
```

The `TenMut` also implements the `CoreMethodsMut` from the same module, this provides some in-place modification and mutable view slicing.

Indexing is provided by [index](flash_powder/src/index.rs), mutably with `i_mut`, this is limited to operations that always produce a view in PyTorch, so indexing with indices is not supported as that returns a copy in some cases.
```rust
let d = Tensor::from(&[
    [1.0f32, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
])?;

let z = d.i((1..3, 0..1))?;  // Equivalent to PyTorch; z = d[1:3, 0:1]
assert_eq!(z.sizes(), &[2, 1]);
let z = d.i((-3isize..3, -3isize..3))?; // z = d[-3:3, -3:3]
assert_eq!(z.sizes(), &[2, 2]); // #PYTHON list(z.shape)
```

The [functional](flash_powder/src/functional.rs) module provides the basic building blocks I needed for vgg & U-net;
`adaptive_avg_pool2d`, `conv2d`, `conv_transpose2d`, `interpolate`, `linear`, `max_pool2d`, `relu` and `upsample`.
This is definitely not fully featured, but shows how to dispatch kernels, they are defined in [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml), which comes with a [README](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/README.md) that explains the flags (permalinks to v2.12, be sure to change that to latest).
Kernel dispatches can of course be done out-of-crate.

Finally, the [nn](flash_powder/src/nn/mod.rs) aims to be the equivalent of [torch.nn](https://docs.pytorch.org/docs/2.12/nn.html).
The [nn::Module](flash_powder/src/nn/module.rs) is the Rust trait equivalent to [torch.nn.Module](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html) and exposes methods like `forward`, `to`, `state_dict`, `load_state_dict`, the [nn::layer](flash_powder/src/nn/layer.rs) module provides the layers for the functions from `functional` that all implement `Module`, as well as `Sequential` to be able to chain type-erased layers together.
The `nn` module also provides some helper functionality around the `StateDict`, which dovetails with the `flash_powder_safetensors` crate to be able to load tensors from disk easily.

### torch_stable
Very minimal (handwritten) bindings for the [LibTorch Stable ABI](https://docs.pytorch.org/docs/2.11/notes/libtorch_stable_abi.html).
This system works through a small set of C functions that provide a limited subset of the functionality from libtorch.

The crate is structured after the [stable](https://github.com/pytorch/pytorch/tree/main/torch/csrc/stable),
[aoti_torch](https://github.com/pytorch/pytorch/tree/main/torch/csrc/inductor/aoti_torch) and [headeronly](https://github.com/pytorch/pytorch/tree/main/torch/headeronly) directories.

There's some support tooling in the contrib submodule, but it's mostly there for testing and superseded by the `flash_powder` crate.

The functionality in this crate is a subset of the upstream functionality, it does not follow Rust lifetimes or safety guarantees.

### flash_powder_safetensors

Helper utilities for working with [safetensors](https://huggingface.co/docs/safetensors/index) are available in the [flash_powder_safetensors](./flash_powder_safetensors) crate.

## Usage

Run this to add the dependency to a cargo project in both `build-dependencies` and `dependencies`;
```
cargo add --git https://github.com/iwanders/flash_powder.git flash_powder  -F v2_13,cuda
cargo add --git https://github.com/iwanders/flash_powder.git flash_powder  -F v2_13,cuda --build
```

Update with
```
cargo update
```

## Testing

I want to ensure that the tensors & function arguments follow conventions from the Python side, so there's a heavy emphasis
of testing all functions against their Python equivalents. The Python code to test against is interwoven with the Rust
code with some helper tooling. Tests should run cleanly in valgrind.

### Python truth

For tests, the equivalent Python PyTorch execution is considered the ground truth and the Rust should should produce the
same values.
To be able to easily create reference values in the tests there's a helper tool in `./util/python_truth.py` that can
execute python code in rusts' comment blocks and update values in the rust tests accordingly.
This ensurse that the equivalent python code is next to the rust code in the unit tests and also facilitates automatic
generation of reference values without manual copy pasting which may introduce errors.

The scope of a particular Python execution is limited to within a (test) function scope;

The following:
```rust
/*
    #|PYTHON
    d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([1,4,4])
    w = torch.tensor([[[1.0, 2.0],[3.0, 4.0]]]).unsqueeze(0)
    r = torch.nn.functional.conv2d(d, w)
*/
```
defines what is considered a Python block, this runs the statements in this block in python and stores their values for
use in the next block(s) (either Rust or Python).

The values are then used with a rust comment like: `// #PYTHON <STATEMENT>`, where `<STATEMENT>` is a single Python statement that will be executed.
This comment is placed after the statement it is applied to, it can apply to both constants and function calls like `assert_eq!`.
With function calls, the last argument is replaced with the ground truth.

```rust
assert_eq!(d.sizes(), &[1, 4, 4]); // #PYTHON list(d.shape)
const GROUND_TRUTH: &[usize] = &[1usize, 4, 4]; // #PYTHON list(d.shape)
assert_eq!(
    d.f32_ref()?,
    &[
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0
    ]
); // #PYTHON list(d.view(-1).tolist())
```

Functionality is limited to integers, floats and 1d arrays, in both reference and (implicit) array form.

By default, the binary processess the entire rust file, it can be constrained to a single test with `--test-case test_flash_power_conv2d` or `--test-case test_flash_power_conv*`.

It automatically calls `rustfmt` to ensure files are always formatted after modification.

```
# Extract the python code;
./util/python_truth.py  extract ./flash_powder/src/native_functions.rs --test-case test_flash_power_conv2d
# Execute the python code;
./util/python_truth.py  execute ./flash_powder/src/native_functions.rs
# Execute & substitute the results, write output to /tmp/foo.rs
./util/python_truth.py  substitute ./flash_powder/src/native_functions.rs -o /tmp/foo.rs
# Execute & substitute into the input file.
./util/python_truth.py  update ./flash_powder/src/native_functions.rs
```

When developing, something like this is usually helpful:
```
./util/python_truth.py  update ./flash_powder/src/functional.rs  && cargo t -- --nocapture
```

### Valgrind
There's some helper tooling in `./util/valgrind` to create suppression files against a C++ binary.
These ensure that we ignore some uninitialised values that valgrind finds in the bowels of LibTorch.

Run with these suppressions using valgrind through the runner;
```
./util/valgrind/valgrind.sh target/debug/deps/torch_stable-5f3b6c1dd8420412
```
