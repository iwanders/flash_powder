// I don't really know how to do error handling, since we only ever get a binary signal if it succeeded or not.
// THere's a check for cuda here: https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/macros.h#L9
// The c++ backtrace in the terminal seems hardcoded;
// https://github.com/pytorch/pytorch/blob/c62f72647d6f1bf86025b5e6dbdbd40e6721daf3/torch/csrc/inductor/aoti_torch/utils.h#L14

/// The result of any Stable Torch call, string holds the error message from LibTorch.
///
/// And that requires a branch atm; <https://github.com/pytorch/pytorch/pull/180135>
pub type StableTorchResult<T> = anyhow::Result<T>;

pub fn get_exception_what() -> String {
    let error_msg_ptr = unsafe { crate::stable::c::torch_exception_get_what_without_backtrace() };

    let error_msg = unsafe { std::ffi::CStr::from_ptr(error_msg_ptr) };
    error_msg
        .to_owned()
        .into_string()
        .unwrap_or("failed to convert error message".to_owned())
}

/// For functions that can fail somewhat gracefully.
///
/// Takes a call like:
/// ```
/// # use torch_stable::stable::tensor::Tensor;
/// # use torch_stable::{unsafe_call_bail, StableTorchResult};
/// # use torch_stable::aoti_torch::*;
/// fn foo() -> StableTorchResult<()> {
///   let r = Tensor::new().unwrap();
///   unsafe_call_bail!(aoti_torch_zero_(r.get())); // returns error on failure.
///   Ok(())
/// }
/// ```
#[macro_export]
macro_rules! unsafe_call_bail {
    ($($tokens:tt)*) => {{
        #[allow(clippy::macro_metavars_in_unsafe)]
        let api_call_result = unsafe {$($tokens)*};
        let code_text = stringify!($($tokens)*);
        if api_call_result == $crate::AOTI_TORCH_FAILURE {
            anyhow::bail!("call failed ({}) at {}:{} ({})", code_text, file!(), line!(), $crate::util::get_exception_what());
        }
    }};
}

/// For functions that cannot return a Error.
///
/// This is used in situations like Drop, or functions where it would lead to questionmark soup
/// like getting the dim() on a guaranteed existing tensor pointer.
#[macro_export]
macro_rules! unsafe_call_panic {
    ($($tokens:tt)*) => {{
        #[allow(clippy::macro_metavars_in_unsafe)]
        let api_call_result = unsafe {$($tokens)*};
        let code_text = stringify!($($tokens)*);
        if api_call_result == $crate::AOTI_TORCH_FAILURE {
            panic!("call failed ({}) at {}:{} ({})", code_text, file!(), line!(), $crate::util::get_exception_what());
        }
    }};
}

/// For dispatching kernels, this is likely the one you need.
///
/// Takes a call like:
/// ```
/// # use torch_stable::stable::tensor::Tensor;
/// # use torch_stable::{unsafe_call_dispatch_bail, StableTorchResult};
/// # use torch_stable::aoti_torch::*;
/// fn foo() -> StableTorchResult<Tensor> {
///   let r = Tensor::new().unwrap();
///   let window : &[usize] = &[5, 5];
///   let mut stack: [StableIValue; 2] = [(&r).into(), window.into()];
///   unsafe_call_dispatch_bail!("aten::view", "", stack.as_mut_slice()); // Returns error on failure.
///   let r: Tensor = stack[0].try_into()?;
///   Ok(r)
/// }
/// ```
#[macro_export]
macro_rules! unsafe_call_dispatch_bail {
    ($op_name:expr, $overload_name:expr, $stack:expr) => {{
        let op_name = std::ffi::CString::new($op_name).expect("CString::new failed");
        let op_name_cstr = op_name.as_ptr();

        let overload_name = std::ffi::CString::new($overload_name).expect("CString::new failed");
        let overload_name_cstr = overload_name.as_ptr();
        let local_stack_ptr = $stack.as_mut_ptr();
        let api_call_result = unsafe {
            $crate::stable::c::torch_call_dispatcher(
                op_name_cstr,
                overload_name_cstr,
                local_stack_ptr,
                $crate::TORCH_ABI_VERSION,
            )
        };
        if api_call_result == $crate::AOTI_TORCH_FAILURE {
            anyhow::bail!(
                "dispatch failed ({}, {}) at {}:{} ({})",
                $op_name,
                $overload_name,
                file!(),
                line!(),
                $crate::util::get_exception_what()
            );
        }
    }};
}

/// Like [`unsafe_call_dispatch_bail`], but then panics instead of returning.
///
/// Use this in situations where error propagation isn't possible.
#[macro_export]
macro_rules! unsafe_call_dispatch_panic {
    ($op_name:expr, $overload_name:expr, $stack:expr) => {{
        let op_name = std::ffi::CString::new($op_name).expect("CString::new failed");
        let op_name_cstr = op_name.as_ptr();

        let overload_name = std::ffi::CString::new($overload_name).expect("CString::new failed");
        let overload_name_cstr = overload_name.as_ptr();

        let api_call_result = unsafe {
            $crate::stable::c::torch_call_dispatcher(
                op_name_cstr,
                overload_name_cstr,
                $stack.as_mut_ptr(),
                $crate::TORCH_ABI_VERSION,
            )
        };
        if api_call_result == $crate::AOTI_TORCH_FAILURE {
            panic!(
                "dispatch failed ({}, {}) at {}:{} ({})",
                $op_name,
                $overload_name,
                file!(),
                line!(),
                $crate::util::get_exception_what()
            );
        }
    }};
}
