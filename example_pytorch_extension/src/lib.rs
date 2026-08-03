use flash_powder as fp;
use flash_powder::Tensor;
use flash_powder::prelude::*;

use torch_stable::aoti_torch::AtenTensorHandle;
use torch_stable::{
    aoti_torch::{
        StableIValue, TorchLibraryHandleWrapper, aoti_torch_library_def, aoti_torch_library_impl,
        aoti_torch_library_init_def,
    },
    stable::tensor::Tensor as StableTensor,
    unsafe_call_panic,
};

// Super janky initialisation function that runs my_init_function when this library is loaded.
#[used]
#[unsafe(link_section = ".init_array")]
static INIT_FUNC: extern "C" fn() = my_init_function;

use std::sync::OnceLock;

static LIBRARY_HANDLE: OnceLock<TorchLibraryHandleWrapper> = OnceLock::new();

extern "C" fn my_init_function() {
    println!("test");

    // Next, we can register a library handle.
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

    // Next, we can def some symbols
    // https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/stable/library.h#L97-L101
    let schema = c"mymuladd(Tensor a, Tensor b) -> Tensor";
    unsafe_call_panic!(aoti_torch_library_def(
        LIBRARY_HANDLE.get().unwrap().0,
        schema.as_ptr(),
    ));

    let schema = c"simple() -> ()";
    unsafe_call_panic!(aoti_torch_library_def(
        LIBRARY_HANDLE.get().unwrap().0,
        schema.as_ptr(),
    ));

    let schema = c"simple_takes_tensor(Tensor a) -> ()";
    unsafe_call_panic!(aoti_torch_library_def(
        LIBRARY_HANDLE.get().unwrap().0,
        schema.as_ptr(),
    ));

    unsafe_call_panic!(aoti_torch_library_def(
        LIBRARY_HANDLE.get().unwrap().0,
        c"simple_returns_tensor() -> (Tensor)".as_ptr(),
    ));
    // Next, we need to actually provide the implementation for it.

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

    // And one that accepts a tensor.

    let name = c"simple_takes_tensor";
    extern "C" fn simple_takes_tensor(stack: *mut StableIValue, num_input: u64, num_outputs: u64) {
        println!(
            "Invoking the simple_takes_tensor inputs:  {num_input:?} outputs: {num_outputs:?} "
        );
        // Take ownership of the input tensor:
        let a_ivalue = unsafe { *stack.offset(0) };
        let a_stable_tensor: StableTensor = a_ivalue.try_into().unwrap();
        let a_fp_tensor: Tensor = Tensor::new(a_stable_tensor);
        // Leaving the scope will destroy it.

        println!("return of simple_takes_tensor now, we will destroy the input tensor.");
        unsafe { *stack.offset(0) = StableIValue(0) }; // clear the stack, it's only prudent.
    }
    unsafe_call_panic!(aoti_torch_library_impl(
        LIBRARY_HANDLE.get().unwrap().0,
        name.as_ptr(),
        simple_takes_tensor,
    ));

    // Next is a function that returns a single tensor.
    extern "C" fn simple_returns_tensor(
        stack: *mut StableIValue,
        num_input: u64,
        num_outputs: u64,
    ) {
        println!(
            "Invoking the simple_returns_tensor inputs:  {num_input:?} outputs: {num_outputs:?} "
        );
        // Create a fp::Tensor, convert it into StableTensor
        let res: fp::Tensor = 3i32.try_into().unwrap();
        let stable_tensor: StableTensor = unsafe { res.into_stable_tensor() };

        // Convert it into the raw pointer, extracting an owning raw pointer.
        let tensor_opaque: AtenTensorHandle = stable_tensor
            .into_inner()
            .map(|a| a.into_raw())
            .expect("should be extractable");

        // Next, we convert that to an StableIValue and assign that into the stack.
        let res_tensor: StableIValue = StableIValue(tensor_opaque as _);
        unsafe { *stack = res_tensor };

        // Nothing left to do, this will return the value.
    }
    unsafe_call_panic!(aoti_torch_library_impl(
        LIBRARY_HANDLE.get().unwrap().0,
        c"simple_returns_tensor".as_ptr(),
        simple_returns_tensor,
    ));

    // Next, do something more complex that takes two input tensors and returns one.
    let name = c"mymuladd";

    extern "C" fn mymuladd_fun(stack: *mut StableIValue, num_input: u64, num_outputs: u64) {
        println!("Invoking the fun inputs:  {num_input:?} outputs: {num_outputs:?} ");

        // Interpret the first stack variable as an tensor.
        let a_ivalue = unsafe { *stack.offset(0) };
        let a_stable_tensor: StableTensor = a_ivalue.try_into().unwrap();
        println!("a_stable_tensor, get: {:?}", a_stable_tensor.get());
        let a_fp_tensor: Tensor = Tensor::new(a_stable_tensor);
        // And the second one.
        let b_ivalue = unsafe { *stack.offset(1) };
        let b_stable_tensor: StableTensor = b_ivalue.try_into().unwrap();
        println!("b_stable_tensor, get: {:?}", b_stable_tensor.get());
        let b_fp_tensor: Tensor = Tensor::new(b_stable_tensor);

        // We finished parsing the stack, clear it, since we own the variables now.
        unsafe { *stack.offset(0) = StableIValue(0) };
        unsafe { *stack.offset(1) = StableIValue(0) };

        // Now we can multiply them.
        let res = a_fp_tensor.mul(&b_fp_tensor).unwrap();

        // Next is converting it to a stable tensor, and extract an owning raw pointer.
        let stable_tensor = unsafe { res.into_stable_tensor() };
        let tensor_opaque: AtenTensorHandle = stable_tensor
            .into_inner()
            .map(|a| a.into_raw())
            .expect("should be extractable");

        // Next we assign this into the stack.
        let res_tensor: StableIValue = StableIValue(tensor_opaque as _);
        unsafe { *stack.offset(0) = res_tensor };
    }
    unsafe_call_panic!(aoti_torch_library_impl(
        LIBRARY_HANDLE.get().unwrap().0,
        name.as_ptr(),
        mymuladd_fun,
    ));
}
