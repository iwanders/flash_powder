#!/usr/bin/env python3

from pathlib import Path
import ctypes

sopath = Path(__file__).parent.parent / "target" / "debug"/ "libexample_pytorch_extension.so"
print(sopath)


# 1. Preferred method: Using the constructor (cross-platform compatible) 
my_lib = ctypes.CDLL(sopath)
#torch.ops.load_library(Path(__file__).parent / "build/lib.linux-x86_64-cpython-313/extension_cpp.abi3.so")
#print(torch.ops.extension_cpp.mymuladd)
