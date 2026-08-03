#!/usr/bin/env python3

from pathlib import Path
import ctypes

raw_load = False
sopath = Path(__file__).parent.parent / "target" / "debug"/ "libexample_pytorch_extension.so"

if raw_load:
    print(sopath)
    my_lib = ctypes.CDLL(sopath)
else: 
    import torch
    torch.ops.load_library(str(sopath))
    print(dir(torch.ops.extension_cpp))
    print(torch.ops.extension_cpp.name)
    print(torch.ops.extension_cpp._dir)
    print(torch.ops.extension_cpp.mymuladd)

    # Call the simple one, this just prints 'test'
    print("torch.ops.extension_cpp.simple")
    r = torch.ops.extension_cpp.simple()
    print()

    # Next, we can pass a tensor to takes simple tensor.
    a = torch.randn((2,2))
    print(f"torch.ops.extension_cpp.simple_takes_tensor with {a}")
    r = torch.ops.extension_cpp.simple_takes_tensor(a)
    print("\n")

    
    print("torch.ops.extension_cpp.simple_returns_tensor")
    r = torch.ops.extension_cpp.simple_returns_tensor()
    print(f"simple returns tensor returned {r}\n")


    
    a = torch.randn((2,2))
    b = torch.randn((2,2))
    
    print("torch.ops.extension_cpp.mymuladd")
    print("a:", a)
    print("b:", b)
    
    r = torch.ops.extension_cpp.mymuladd(a,b)
    print(f"r: {r}\n")
