//! Holds the three Tensor types.
use crate::StableTorchResult;
use crate::core_methods::CoreMethods;
use crate::{DType, Device, Layout};
use anyhow;
use torch_stable::aoti_torch::AtenTensorHandle;
use torch_stable::stable::tensor::Tensor as StableTensor;
use torch_stable::unsafe_call_bail;

/// A tensor, this owns its data.
///
/// Interact with it through any of the traits that are implemented for [`TensorAccess`].
///
/// Usually you don't create this directly, but create tensors through [`crate::factory::TensorFactory`].
pub struct Tensor {
    tensor: StableTensor,
}
impl Clone for Tensor {
    /// This is a full owning clone, but lazy, it only materializes when either the source or destination is written to.
    ///
    /// Under the hood this calls <https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1278>, so the `_lazy_clone` kernel.
    ///
    /// The docs for that function state:
    ///
    /// > Like clone, but the copy takes place lazily, only if either the input or the output are written.
    ///
    /// Since clone can't fail in rust, I chose this because a lazy clone is unlikely to cause an out of memory error.
    ///
    /// It does mean that memory allocation errors are deffered to later in the program, but hopefully they can be handled there.
    ///
    /// This does not work for Ten's that are borrowed from byte slices through from blob.
    ///
    fn clone(&self) -> Self {
        // Clone cannot throw... so we use a lazy clone; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1278
        self.lazy_clone().unwrap()
    }
}

impl Tensor {
    /// Create a new tensor backed by the provided StableTensor.
    ///
    /// The provided tensor should be detached from anything else and exclusive ownership should be passed.
    pub fn new(tensor: StableTensor) -> Self {
        Self { tensor }
    }

    /// Equivalent to torch.tensor(data)
    ///
    /// Always allocates in the provided data type, on the cpu.
    ///
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.tensor.html#torch.tensor>)
    ///
    /// Is actually implemented via TryInto
    pub fn from<T>(data: T) -> StableTorchResult<Tensor>
    where
        T: TryInto<Tensor>,
        T::Error: Into<anyhow::Error>,
    {
        let b: StableTorchResult<Tensor> = data.try_into().map_err(|e| e.into());
        b
    }
}

/// A borrow on another Tensor, like a view into one.
pub struct Ten<'a> {
    // This is the backing tensor that shares data with the 'parent'.
    tensor: StableTensor,
    parent: std::marker::PhantomData<&'a ()>,
}

// https://github.com/pytorch/pytorch/blob/01d9abd0bb0eeea5416b0ceb75d243362cc90aee/torch/csrc/stable/ops.h#L727-L811
pub type BlobDeleter = fn(*mut std::ffi::c_void, *mut std::ffi::c_void);
#[derive(Copy, Clone, Debug)]
pub struct BlobOptionsBytes<'b> {
    /// The size of each dimension (in DType units), array length must match strides.
    pub sizes: &'b [usize],
    /// The stride of each dimension (in DType units), array length must match sizes.
    pub strides: &'b [usize],
    /// The data type represented by the bytes.
    pub dtype: DType,
    // Layout is usually strided.
    // pub layout: Layout,
    // I don't think in Rust we can get a &[u8] that is not on the cpu side
    // /// The device this data is on.
    // pub device: Device,
}
impl<'a> Ten<'a> {
    pub fn new(parent: std::marker::PhantomData<&'a ()>, tensor: StableTensor) -> Self {
        Self { parent, tensor }
    }
    pub(crate) fn as_parent(&self) -> std::marker::PhantomData<&'a ()> {
        self.parent
    }

    pub fn to_owned(&self) -> StableTorchResult<Tensor> {
        self.to_tensor()
    }

    /// Create a view of a tensor with data provided by the slice.
    ///
    /// From the docs:
    /// > Creates a tensor that uses the provided data pointer as its storage. The tensor does not own the data, so the caller must ensure the data remains valid for the lifetime of the tensor.
    ///
    /// This Ten<'d> does not actually have a storage pointer, as such is cannot be lazily cloned and MUST be cloned with  [`CoreMethods::to_tensor`][`crate::core_methods::CoreMethods::to_tensor`].
    pub fn from_bytes<'d, 'b>(
        data: &'d [u8],
        options: &BlobOptionsBytes<'b>,
    ) -> StableTorchResult<Ten<'d>> {
        use std::mem::transmute;
        /*
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
             deleter: BlobDeleter, // void (*deleter)(void* data, void* ctx),
             deleter_ctx: *mut c_void,
         ) -> AOTITorchError;
        */
        let data_ptr: *const u8 = data.as_ptr();
        let data_void: *mut std::ffi::c_void = unsafe { transmute(data_ptr) };
        dbg!(data_ptr, data_void, data.len());
        if options.strides.len() != options.sizes.len() {
            anyhow::bail!("strides and sizes should be equal length");
        }
        let options_scalartype: torch_stable::headeronly::core::ScalarType = options.dtype.into();
        let element_size = unsafe {
            torch_stable::aoti_torch::aoti_torch_dtype_element_size(options_scalartype as _)
        };
        dbg!(options.dtype);
        dbg!(element_size);
        let last_position: usize = options
            .sizes
            .iter()
            .zip(options.strides.iter())
            .map(|(size, stride)| (size - 1) * stride)
            .sum();
        let last_byte = (last_position * element_size) + element_size;
        dbg!(last_byte);
        if data.len() < last_byte {
            anyhow::bail!(
                "the provided data length is not sufficient to read the last element at {last_position} of {element_size} bytes"
            );
        }
        let ndim: i64 = options.strides.len() as _;
        let sizes_ptr: *const i64 = unsafe { transmute(options.sizes.as_ptr()) };
        let strides_ptr: *const i64 = unsafe { transmute(options.strides.as_ptr()) };
        let storage_offset = 0;
        let dtype: i32 = options_scalartype as _;
        let device = Device::CPU;
        let device_type: i32 = device.device_type() as _;
        let device_index: i32 = device.device_index().0;
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        let layout: i32 = Layout::Strided as _;
        let opaque_metadata: *const u8 = std::ptr::null();
        let opaque_metadata_size: i64 = 0;
        let deleter = None;
        let deleter_ctx: *mut std::ffi::c_void = std::ptr::null_mut();

        // With all the prep done, we can finally invoke the monster!
        unsafe_call_bail!(torch_stable::stable::c::torch_from_blob(
            data_void,
            ndim,
            sizes_ptr,
            strides_ptr,
            storage_offset,
            dtype,
            device_type,
            device_index,
            &mut handle_res,
            layout,
            opaque_metadata,
            opaque_metadata_size,
            deleter,
            deleter_ctx,
        ));

        let marker = std::marker::PhantomData::<&'d ()>;
        Ok(Ten::new(marker, StableTensor::from_handle(handle_res)))
    }
}

/// A mutable borrow on another Tensor, like mutably borrowed slice into one.
pub struct TenMut<'a> {
    // This is the backing tensor that shares data with the 'parent'.
    tensor: StableTensor,
    _parent: &'a mut StableTensor,
}
impl<'a> TenMut<'a> {
    pub fn new(parent: &'a mut StableTensor, tensor: StableTensor) -> Self {
        Self {
            _parent: parent,
            tensor,
        }
    }
    pub(crate) fn into_parent(self) -> &'a mut StableTensor {
        self._parent
    }
}

/// Constant tensor access.
pub trait TensorAccess {
    fn get_tensor(&self) -> &StableTensor;
}

/// Mutable tensor access
pub trait TensorAccessMut {
    fn get_tensor_mut(&mut self) -> &mut StableTensor;
}

impl<'a> TensorAccess for TenMut<'a> {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
}

impl<'a> TensorAccessMut for TenMut<'a> {
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        &mut self.tensor
    }
}

impl<'a> TensorAccess for Ten<'a> {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
}

impl TensorAccess for Tensor {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
}

impl TensorAccessMut for Tensor {
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        &mut self.tensor
    }
}

impl TensorAccess for &Tensor {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
}

impl TensorAccessMut for &mut Tensor {
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        &mut self.tensor
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_flash_powder_ten_from_blob() -> StableTorchResult<()> {
        use crate::prelude::*;
        let d = Tensor::from(&[[1.0f32, 2.0], [3.0, 4.0]])?;

        let data = d.data()?;
        let sizes = d.sizes();
        assert_eq!(sizes, &[2, 2]);
        let strides = d.strides();
        assert_eq!(strides, &[2, 1]);
        let options = BlobOptionsBytes {
            sizes: sizes,
            strides: strides,
            dtype: d.dtype(),
        };
        assert_eq!(options.dtype, DType::F32);

        let ten_thing = Ten::from_bytes(data, &options)?;
        assert!(d.is_equal(&ten_thing)?);

        let u8_3x2 = Tensor::from(&[[1u8, 2], [3, 4], [5, 6]])?;
        println!("tensor: {:?}, shape: {:?}", u8_3x2, u8_3x2.shape());
        println!(
            "sizes: {:?}, strides: {:?}",
            u8_3x2.sizes(),
            u8_3x2.strides()
        );

        let data = [1, 2, 3, 4, 5, 6u8];
        let sizes = &[3, 2];
        let strides = &[2, 1];
        let options = BlobOptionsBytes {
            sizes: sizes,
            strides: strides,
            dtype: DType::U8,
        };

        let ten_3x2x1 = Ten::from_bytes(&data, &options)?;
        assert_eq!(&ten_3x2x1.shape(), &[3, 2]);

        let u32_3x2 = Tensor::from(&[[1u32, 2], [3, 4], [5, 6]])?;
        println!("tensor: {:?}, shape: {:?}", u32_3x2, u32_3x2.shape());
        println!(
            "sizes: {:?}, strides: {:?}",
            u32_3x2.sizes(),
            u32_3x2.strides()
        );
        let data = u32_3x2.data()?;
        println!("data: {data:?}");
        let sizes = &[3, 2];
        let strides = &[2, 1];
        let options = BlobOptionsBytes {
            sizes: sizes,
            strides: strides,
            dtype: DType::U32,
        };
        let ten_u16_3x2 = Ten::from_bytes(&data, &options)?;
        println!(
            "ten_u16_3x2: {:?}, shape: {:?}",
            ten_u16_3x2,
            ten_u16_3x2.shape()
        );

        assert_eq!(&ten_u16_3x2.shape(), &[3, 2]);
        assert!(ten_u16_3x2.is_equal(&u32_3x2)?);

        Ok(())
    }
}
