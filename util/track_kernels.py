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


print("doing the plus assign")
@torch.compile
def plus_assign_v(left, right):
    left += right
    # heh, this doesn't result in a single kernel, instead;
    # Topologically Sorted Source Nodes: [left], Original ATen: [aten.add, aten.copy_]
    #triton_poi_fused_add_copy__0.run(arg0_1, arg1_1, arg0_1, 4, stream=raw_stream0)
    # Which pretty much does;
    # tmp0 = load left
    # tmp1 = load right
    # tmp2 = tmp0 + tmp1
    # store(left, tmp2)



a = torch.tensor([(1.0, 0.0), (0.0, 1.0)])
b = torch.tensor([(2.0, 1.0), (1.0, 2.0)])
# Using cuda here ensures we actually see a kernel.
r = plus_assign_v(a.to("cuda"), b.to("cuda"))
print(r)
