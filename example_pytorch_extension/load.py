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

     
    r = torch.ops.extension_cpp.simple()
    a = torch.randn((2,2))
    r = torch.ops.extension_cpp.simple_takes_tensor( a)
    print(a) # Up to here is good.
    r = torch.ops.extension_cpp.simple_returns_tensor()
    print(r) # Good now!


    
    a = torch.randn((2,2))
    b = torch.randn((2,2))
    # c = 1.0
    print("a:", a)
    print("b:", b)
    
    r = torch.ops.extension_cpp.mymuladd(a,b)
    
    print("r:", r)
