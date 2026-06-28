#!/usr/bin/env python3

import os

os.environ["TORCH_LOGS"] = "output_code" 
import torch
 
color_lookup = torch.tensor([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])

the_indices = torch.tensor([0, 1, 2, 2, 1, 0, 1,2,0], dtype=torch.long)


@torch.compile
def my_index_function(color_lookup, the_indices):
    x = color_lookup[the_indices]
    #buf0 = torch.ops.aten.index.Tensor(arg0_1, [arg1_1])
    return x

# Using cuda here ensures we actually see a kernel.
r = my_index_function(color_lookup.to("cuda"), the_indices)
print(r)
r[0,0] = 3.0
print(r)
print(color_lookup)
