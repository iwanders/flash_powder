use super::super::super::aoti_torch::*;
use libc::{c_char, c_void};

// https://github.com/pytorch/pytorch/blob/b5afee0dda86ab028efe66c1eac7028a351edc90/torch/csrc/stable/c/shim.h#L69

// Keep the order the same as the original file.
unsafe extern "C" {
    // https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/c/shim.h#L20-L29
    pub unsafe fn torch_call_dispatcher(
        op_name: *const c_char,
        overload_name: *const c_char,
        stack: *const StableIValue,
        extension_build_version: u64,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/c/shim.h#L69
    pub unsafe fn torch_parse_device_string(
        device_string: *const c_char,
        out_device_type: *mut u32,
        out_device_index: *mut i32,
    ) -> AOTITorchError;

}
// https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/c/shim.h#L39
#[repr(C)]
pub struct StableListOpaque {
    _private: [u8; 0],
}
pub type StableListHandle = *mut StableListOpaque;

unsafe extern "C" {

    // returns an owning reference of a StableList. callee is responsible for
    // freeing memory.
    pub unsafe fn torch_new_list_reserve_size(
        size: usize,
        ret: *mut StableListHandle,
    ) -> AOTITorchError;

    pub unsafe fn torch_list_size(
        list_handle: StableListHandle,
        size: *mut usize,
    ) -> AOTITorchError;

    pub unsafe fn torch_list_get_item(
        list_handle: StableListHandle,
        index: usize,
        element: *mut StableIValue,
    ) -> AOTITorchError;

    pub unsafe fn torch_list_set_item(
        list_handle: StableListHandle,
        index: usize,
        element: StableIValue,
    ) -> AOTITorchError;

    pub unsafe fn torch_list_push_back(
        list_handle: StableListHandle,
        element: StableIValue,
    ) -> AOTITorchError;

    // deletes the underlying list referenced by list_handle
    pub unsafe fn torch_delete_list(list_handle: StableListHandle) -> AOTITorchError;
}

unsafe extern "C" {

    // https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/c/shim.h#L94
    // Get a pointer to the underlying storage data
    pub unsafe fn torch_get_mutable_data_ptr(
        tensor: AtenTensorHandle,
        ret_data_ptr: *mut *mut c_void, // returns borrowed reference
    ) -> AOTITorchError;

    pub unsafe fn torch_get_const_data_ptr(
        tensor: AtenTensorHandle,
        ret_data_ptr: *mut *const c_void, // returns borrowed reference
    ) -> AOTITorchError;
}

// From https://github.com/pytorch/pytorch/pull/180135/
#[cfg(feature = "v2_13")]
unsafe extern "C" {

    pub unsafe fn torch_exception_get_what() -> *const c_char;
    pub unsafe fn torch_exception_get_what_without_backtrace() -> *const c_char;

}

// https://github.com/pytorch/pytorch/pull/179421
#[cfg(feature = "v2_13")]
unsafe extern "C" {

    pub unsafe fn torch_new_stable_ivalue(
        allocated_value: *mut *mut StableIValue,
    ) -> AOTITorchError;
    pub unsafe fn torch_delete_stable_ivalue(value_to_delete: *mut StableIValue) -> AOTITorchError;

}

pub type BlobDeleter = extern "C" fn(*mut c_void /* data */, *mut c_void /* ctx */);
// https://github.com/pytorch/pytorch/blob/01d9abd0bb0eeea5416b0ceb75d243362cc90aee/torch/csrc/stable/c/shim.h#L167-L187
// https://github.com/pytorch/pytorch/blob/01d9abd0bb0eeea5416b0ceb75d243362cc90aee/torch/csrc/stable/ops.h#L727-L811
unsafe extern "C" {
    pub unsafe fn torch_from_blob(
        data: *mut c_void,
        ndim: i64,
        sizes_ptr: *const i64,
        strides_ptr: *const i64,
        storage_offset: i64,
        dtype: i32,
        device_type: i32,
        device_index: i32,
        ret: &mut AtenTensorHandle,
        layout: i32,
        opaque_metadata: *const u8,
        opaque_metadata_size: i64,
        deleter: Option<BlobDeleter>, // void (*deleter)(void* data, void* ctx),
        deleter_ctx: *mut c_void,
    ) -> AOTITorchError;

}
