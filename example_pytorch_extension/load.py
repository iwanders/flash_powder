#!/usr/bin/env python3

import ctypes
from pathlib import Path

sopath = Path(__file__).parent.parent / "target" / "debug"/ "libexample_pytorch_extension.so"

# Use this to test whether the initialisation actually runs.
if False:
    import sys
    my_lib = ctypes.CDLL(sopath)
    sys.exit(0)


import torch  # pyright: ignore[reportMissingImports]

torch.ops.load_library(str(sopath))
print(dir(torch.ops.extension_from_rust))
print(torch.ops.extension_from_rust.name)
print(torch.ops.extension_from_rust._dir)
print(torch.ops.extension_from_rust.mymuladd)

# Call the simple one, this just prints 'test'
print("torch.ops.extension_from_rust.simple")
r = torch.ops.extension_from_rust.simple()
print()

# Next, we can pass a tensor to takes simple tensor.
a = torch.randn((2,2))
print(f"torch.ops.extension_from_rust.simple_takes_tensor with {a}")
r = torch.ops.extension_from_rust.simple_takes_tensor(a)
print("\n")


print("torch.ops.extension_from_rust.simple_returns_tensor")
r = torch.ops.extension_from_rust.simple_returns_tensor()
print(f"simple returns tensor returned {r} with repr: {repr(r)}\n")
assert isinstance(r, torch.Tensor)



a = torch.randn((2,2))
b = torch.randn((2,2))

print("torch.ops.extension_from_rust.mymuladd")
print("a:", a)
print("b:", b)

r = torch.ops.extension_from_rust.mymuladd(a,b)
print(f"r: {r}\n")
