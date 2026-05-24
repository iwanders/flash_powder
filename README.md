## Flash Powder

> What makes light and works through oxidization? Flash Powder.


A very minimal rust wrapper for libtorch, using the [Torch Stable API](https://docs.pytorch.org/cppdocs/stable.html) only.
This is mostly my project to gain a better understanding of how (lib/py)torch works under the hood. I do not recommend using this.

The stable ABI doesn't expose all functionality of libtorch, but a surprising amount of functionality is available,
especially if the goal is just to do inference. The [example_vgg](./example_vgg) crate holds an implementation of vgg11.

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

### torch_stable
Very minimal (handwritten) bindings for the [LibTorch Stable ABI](https://docs.pytorch.org/docs/2.11/notes/libtorch_stable_abi.html).
This system works through a small set of C functions that provide a limited subset of the functionality from libtorch.

The crate is structured after the [stable](https://github.com/pytorch/pytorch/tree/main/torch/csrc/stable),
[aoti_torch](https://github.com/pytorch/pytorch/tree/main/torch/csrc/inductor/aoti_torch) and [headeronly](https://github.com/pytorch/pytorch/tree/main/torch/headeronly) directories.

There's some support tooling in the contrib submodule, but it's mostly there for testing and superseded by the `flash_powder` crate.

The functionality in this crate is a subset of the upstream functionality, it does not follow Rust lifetimes or safety guarantees.

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
