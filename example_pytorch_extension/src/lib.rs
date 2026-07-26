// Quick example to test if we can make a pytorch extension.
//
// There's a lot macro stuff here:
// https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L2

// So the entry appears to be that STABLE_TORCH_LIBRARY(extension_cpp, m) entry.
// https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L355
// Which seems to just instantiate this class; https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L130-L146
// WHich in turn calls the initFn on load. We an to the same.

use flash_powder as fp;
use flash_powder::Tensor;
use flash_powder::prelude::*;

#[used]
#[unsafe(link_section = ".init_array")]
static INIT_FUNC: extern "C" fn() = my_init_function;

use torch_stable::aoti_torch::AtenTensorHandle;
use torch_stable::{
    aoti_torch::{
        StableIValue, TorchLibraryHandleWrapper, aoti_torch_library_def, aoti_torch_library_impl,
        aoti_torch_library_init_def,
    },
    unsafe_call_panic,
};

extern "C" fn my_init_function() {
    // Your pre-main initialization logic here
    println!("test");
    unsafe {
        let mut handle_res: TorchLibraryHandleWrapper = TorchLibraryHandleWrapper::new_null();
        let ns = c"extension_cpp";
        let file = c"thisfile";
        unsafe_call_panic!(aoti_torch_library_init_def(
            ns.as_ptr(),
            file.as_ptr(),
            1,
            &mut handle_res.0
        ));
        println!("handle: {handle_res:?}");
        LIBRARY_HANDLE
            .set(handle_res)
            .expect("should be able to set it, this is called once");
    }

    // Next, we can def a symbol.
    // https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L97-L101
    /*
     * STABLE_TORCH_LIBRARY(extension_cpp, m) {
       // Note that "float" in the schema corresponds to the C++ double type
       // and the Python float type.
       m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
     }

    */
    let schema = c"mymuladd(Tensor a, Tensor b, float c) -> Tensor";
    unsafe_call_panic!(aoti_torch_library_def(
        LIBRARY_HANDLE.get().unwrap().0,
        schema.as_ptr(),
    ));
    let schema = c"simple() -> ()";
    unsafe_call_panic!(aoti_torch_library_def(
        LIBRARY_HANDLE.get().unwrap().0,
        schema.as_ptr(),
    ));

    // Next, we need to actually provide the implementation for it.
    /*
    STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
      m.impl("mymuladd", TORCH_BOX(&mymuladd_cpu));
    }
    */
    // https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L63-L95
    // Oh, we should probably use the https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L63-L95 flavour instead
    // lets skip that for now.
    // Not sure what the whole purpose of the boxing is here; https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L284-L328
    // Maybe it's just to collect the signature automatically?

    // https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L63-L84

    let name = c"simple";
    extern "C" fn fun_simple(stack: *mut StableIValue, num_input: u64, num_outputs: u64) {
        println!("Invoking the fun_simple inputs:  {num_input:?} outputs: {num_outputs:?} ");
    }
    unsafe_call_panic!(aoti_torch_library_impl(
        LIBRARY_HANDLE.get().unwrap().0,
        name.as_ptr(),
        fun_simple,
    ));
    //
    let name = c"mymuladd";

    extern "C" fn fun(stack: *mut StableIValue, num_input: u64, num_outputs: u64) {
        println!("Invoking the fun inputs:  {num_input:?} outputs: {num_outputs:?} ");
        let a_ivalue = unsafe { *stack.offset(0) };
        let a_stable_tensor: torch_stable::stable::tensor::Tensor = a_ivalue.try_into().unwrap();
        let a_fp_tensor: Tensor = Tensor::new(a_stable_tensor);
        let b_ivalue = unsafe { *stack.offset(1) };
        let b_stable_tensor: torch_stable::stable::tensor::Tensor = b_ivalue.try_into().unwrap();
        let b_fp_tensor: Tensor = Tensor::new(b_stable_tensor);

        let res = a_fp_tensor.add(&b_fp_tensor).unwrap();
        println!("res: {res:?}");

        // Leak both tensors tensors that we wrapped...?
        unsafe {
            a_fp_tensor
                .into_stable_tensor()
                .into_inner()
                .map(|a| a.into_raw());
        }
        unsafe {
            b_fp_tensor
                .into_stable_tensor()
                .into_inner()
                .map(|a| a.into_raw());
        }

        let res: Tensor = 3i32.try_into().unwrap();
        // Next, we need to assign the result back into the stack.
        let stable_tensor = unsafe { res.into_stable_tensor() };
        // Oh we probably have to leak it here...
        let tensor_opaque: AtenTensorHandle = stable_tensor
            .into_inner()
            .map(|a| a.into_raw())
            .expect("should be extractable");

        let res_tensor: StableIValue = StableIValue(tensor_opaque as _);
        unsafe { *stack.offset(0) = res_tensor };
        unsafe { *stack.offset(1) = StableIValue(0) };
        unsafe { *stack.offset(2) = StableIValue(0) };
        unsafe { *stack.offset(3) = StableIValue(0) };
        /*let mut stack: [StableIValue; 3] = [
            (self.get_tensor()).into(),
            other.get_tensor().into(),
            (&string).into(),
        ];
        unsafe_call_dispatch_panic!("aten::div", "Tensor_mode", stack.as_mut_slice());*/
    }
    unsafe_call_panic!(aoti_torch_library_impl(
        LIBRARY_HANDLE.get().unwrap().0,
        name.as_ptr(),
        fun,
    ));
}
use std::sync::OnceLock;

static LIBRARY_HANDLE: OnceLock<TorchLibraryHandleWrapper> = OnceLock::new();
