
Registration of the extension is working, it can be loaded, functions can be ran.

Passing tensors back and forth is not yet working, according to [the impl documentation](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L63-L84) we need to own the memory and free it, but that results in memory corruption.

Could it be that the 'boxed' aspect of the function comes into play? Or is that 'box' just syntactic sugar on the c++ side?
