
Errors with:
```
    [W827 19:47:26.609869767 engine.cpp:1307] Warning: Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak. (function operator())

    Error: dispatch failed (aten::_backward, ) at experiments/backward/src/lib.rs:53 (derivative for aten::_foreach_addcmul is not implemented)

```

Because the current addition operation doesn't have a derivative implementation.

But I also don't see a way to retrieve the `.grad()` tensor...
