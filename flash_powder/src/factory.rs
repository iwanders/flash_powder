//! Factory methods to create [`Tensor`].
//!
//! Pytorch puts these in the module, as torch.zeros(), I chose to put them as static methods on Tensor.

use crate::{StableTorchResult, Tensor, TensorAccess, dtype::DType, Ten};
use torch_stable::aoti_torch::{aoti_torch_zero_, AtenTensorHandle};
use torch_stable::headeronly::core::{Layout, MemoryFormat};
use torch_stable::stable::device::Device;
use torch_stable::{
    aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor, unsafe_call_bail,
    unsafe_call_dispatch_bail,
};


/// Options for the `to` operation.
///
/// The types [`Device`], [`DType`], [`Layout`] and [`MemoryFormat`] implement [`std::convert::From`] for this struct.
///
/// This means that you can do:
/// ```rust
/// # use flash_powder::prelude::*;
/// # use flash_powder::{StableTorchResult, Tensor};
/// # use flash_powder as fp;
/// # fn foo() -> StableTorchResult<()>{
///   let t = Tensor::zeros(&[3,3], &fp::DType::U8.into())?;
/// # Ok(())
/// # }
/// ```
/// If you want to populate two fields at the same time you still need to create the struct manually.

#[derive(Copy, Clone, Debug, Default)]
pub struct ToOptions {
    pub dtype: Option<DType>,
    pub layout: Option<Layout>,
    pub device: Option<Device>,
    pub pin_memory: Option<bool>,
    pub memory_format: Option<MemoryFormat>,
    pub non_blocking: bool,
    pub copy: bool,
}


macro_rules! impl_conversion {
    ($t:ty, $dest:ty, $v:ident) => {
        impl std::convert::From<$t> for $dest {
            fn from(value: $t) -> Self {
                Self{$v: Some(value), ..Default::default()}
            }
        }
    };
}
impl_conversion!(Device, ToOptions, device);
impl_conversion!(DType, ToOptions, dtype);
impl_conversion!(Layout, ToOptions, layout);
impl_conversion!(MemoryFormat, ToOptions, memory_format);



/// Options for empty.
///
/// The types [`Device`], [`DType`], [`Layout`] and [`MemoryFormat`] implement [`std::convert::From`] for this struct.
#[derive(Copy, Clone, Debug, Default)]
pub struct EmptyOptions {
    pub dtype: Option<DType>,
    pub layout: Option<Layout>,
    pub device: Option<Device>,
    pub pin_memory: Option<bool>,
    pub memory_format: Option<MemoryFormat>,
}
impl_conversion!(Device, EmptyOptions, device);
impl_conversion!(DType, EmptyOptions, dtype);
impl_conversion!(Layout, EmptyOptions, layout);
impl_conversion!(MemoryFormat, EmptyOptions, memory_format);


/// Options to create various tensors.
///
/// The types [`Device`], [`DType`], [`Layout`]  implement [`std::convert::From`] for this struct.
#[derive(Copy, Clone, Debug, Default)]
pub struct TensorOptions {
    pub dtype: Option<DType>,
    pub layout: Option<Layout>,
    pub device: Option<Device>,
    pub pin_memory: Option<bool>,
}
impl_conversion!(Device, TensorOptions, device);
impl_conversion!(DType, TensorOptions, dtype);
impl_conversion!(Layout, TensorOptions, layout);

/// Native functions that produce owned tensors.
///
/// See the [`factory`][crate::factory] module for description of this trait's functionality.
/// This trait is only implemented for [`Tensor`].
///
/// ```
///   # use flash_powder::prelude::*;
///   # use flash_powder::Tensor;
///   let a = Tensor::empty(&[5, 5], &Default::default()).unwrap();
///   assert_eq!(a.sizes(), &[5, 5]);
/// ```
pub trait TensorFactory {
    /// A new empty vector
    ///
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L2425)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.empty.html#torch.empty)
    ///
    fn empty(dimensions: &[usize], options: &EmptyOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 6] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
            (&options.memory_format).into(),
        ];
        // https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L2424
        unsafe_call_dispatch_bail!("aten::empty", "memory_format", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        unsafe_call_bail!(aoti_torch_zero_(r.get()));

        Ok(Tensor::new(r))
    }
    /// A new zeros vector
    ///
    ///
    ///
    /// ```rust
    /// # use flash_powder::prelude::*;
    /// # use flash_powder::{StableTorchResult, Tensor};
    /// # use flash_powder as fp;
    /// # fn foo() -> StableTorchResult<()>{
    ///   let t = Tensor::zeros(&[3,3], &fp::DType::U8.into())?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L6837)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.zeros.html)
    ///
    //
    // https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L6800
    fn zeros(dimensions: &[usize], options: &TensorOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 5] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
        ];
        unsafe_call_dispatch_bail!("aten::zeros", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(Tensor::new(r))
    }

    /// A new ones vector
    ///
    ///
    ///
    /// ```rust
    /// # use flash_powder::prelude::*;
    /// # use flash_powder::{StableTorchResult, Tensor};
    /// # use flash_powder as fp;
    /// # fn foo() -> StableTorchResult<()>{
    ///   let t = Tensor::ones(&[3,3], &fp::DType::U8.into())?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L4621)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.ones.html)
    ///
    //
    // https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L6800
    fn ones(dimensions: &[usize], options: &TensorOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 5] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
        ];
        unsafe_call_dispatch_bail!("aten::ones", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(Tensor::new(r))
    }

    /// A new randn tensor
    ///
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4963)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.randn.html)
    ///
    fn randn(dimensions: &[usize], options: &TensorOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 5] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
        ];
        unsafe_call_dispatch_bail!("aten::randn", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(Tensor::new(r))
    }

    fn from_f32(value: f32) -> StableTorchResult<Tensor> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(
            torch_stable::aoti_torch::aoti_torch_scalar_to_tensor_float32(value, &mut handle_res)
        );
        Ok(Tensor::new(StableTensor::from_handle(handle_res)))
    }

    /// Concatenates the given sequence of tensors in tensors in the given dimension
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L1433)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.cat.html)
    fn cat<T>(tensors: &[&T], dim: usize) -> StableTorchResult<Tensor>
    where
        T: TensorAccess,
    {
        let mut stack: [StableIValue; 2] =
            [tensors.iter().map(|z| z.get_tensor()).collect(), dim.into()];
        unsafe_call_dispatch_bail!("aten::cat", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(Tensor::new(r))
    }
}
impl TensorFactory for Tensor {}


// https://github.com/pytorch/pytorch/blob/01d9abd0bb0eeea5416b0ceb75d243362cc90aee/torch/csrc/stable/ops.h#L727-L811
pub type BlobDeleter =  fn(*mut std::ffi::c_void, *mut std::ffi::c_void);
#[derive(Copy, Clone, Debug )]
pub struct BlobOptionsBytes< 'b> {
    pub sizes: &'b [usize],
    pub strides: &'b [usize],
    pub dtype: DType,
    // Layout is usually strided.
    // pub layout: Layout,
    pub device: Device,
}

pub trait TensorBorrowFactory {

    /// Creates a tensor that uses the provided data pointer as its storage.
    /// The tensor does not own the data, so the caller must ensure the data
    /// remains valid for the lifetime of the tensor.
    fn from_bytes<'d, 'b>(data: &'d [u8], options: &BlobOptionsBytes<  'b>) -> StableTorchResult<Ten<'d>>;
}

impl<'c> TensorBorrowFactory for Ten<'c> {
    fn from_bytes<'d, 'b>(data: &'d [u8], options: &BlobOptionsBytes< 'b>) -> StableTorchResult<Ten<'d>> {
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
        let data_ptr : *const u8 = data.as_ptr();
        let data_void: *mut std::ffi::c_void = unsafe{ transmute(data_ptr)};
        if options.strides.len() != options.sizes.len() {
            anyhow::bail!("strides and sizes should be equal length");
        }

        let element_size = unsafe{torch_stable::aoti_torch::aoti_torch_dtype_element_size(options.dtype as _)};
        let last_position: usize = options.sizes.iter().zip(options.strides.iter()).map(|(size, stride)| (size - 1) * stride).sum();
        let last_byte = last_position + element_size;
        if data.len() < last_byte {
            anyhow::bail!("the provided data length is not sufficient to read the last element at {last_position} of {element_size} bytes");
        }


        let ndim: i64 = options.strides.len() as _;
        let sizes_ptr: *const i64 = unsafe{ transmute(options.sizes.as_ptr())};
        let strides_ptr: *const i64 = unsafe{ transmute(options.strides.as_ptr())};
        let storage_offset = 0;
        let dtype: i32 = options.dtype as _;
        let device_type: i32 = options.device.device_type() as _;
        let device_index : i32 = options.device.device_index().0;
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        let layout : i32 = Layout::Strided as _;
        let opaque_metadata : *const u8 = std::ptr::null();
        let opaque_metadata_size : i64 = 0;
        let deleter  = None;
        let deleter_ctx: *mut std::ffi::c_void = std::ptr::null_mut();

        // With all the prep done, we can finally invoke the monster!
        unsafe_call_bail!(
            torch_stable::stable::c::torch_from_blob(
                data_void ,
                ndim ,
                sizes_ptr ,
                strides_ptr ,
                storage_offset ,
                dtype ,
                device_type ,
                device_index ,
                &mut handle_res,
                layout ,
                opaque_metadata ,
                opaque_metadata_size ,
                deleter ,
                deleter_ctx ,
            )
        );

        // Ok(Ten::new( , StableTensor::from_handle(handle_res)))
        let marker = std::marker::PhantomData::<&'d ()>::default();
        Ok(Ten::new(marker, StableTensor::from_handle(handle_res)))
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_randn() -> StableTorchResult<()> {
        let d = Tensor::randn(&[1000, 1000], &Default::default())?;
        assert_eq!(d.sizes(), &[1000, 1000]);

        let mean = d.mean(&Default::default())?;
        let value = mean.f32s_ref()?[0];
        assert!(value.abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_flash_powder_cat() -> StableTorchResult<()> {
        /*
            #|PYTHON
            x = torch.tensor([[1.0, 2.0],[3.0, 4.0]], dtype=torch.float)
        */

        let d = Tensor::from(&[[1.0f32, 2.0], [3.0, 4.0]])?;
        assert_eq!(d.sizes(), &[2, 2]); // #PYTHON list(x.shape)
        assert_eq!(d.f32s_ref()?, &[1.0f32, 2.0, 3.0, 4.0]); // #PYTHON list(x.view(-1).tolist())

        /*
            #|PYTHON
            a = torch.cat([x,x,x], 0)
        */
        let a = Tensor::cat(&[&d, &d, &d], 0)?;
        assert_eq!(a.sizes(), &[6, 2]); // #PYTHON list(a.shape)
        assert_eq!(
            a.f32s_ref()?,
            &[1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        ); // #PYTHON list(a.view(-1).tolist())
           /*
               #|PYTHON
               b = torch.cat([x,x,x], 1)
           */
        let b = Tensor::cat(&[&d, &d, &d], 1)?;
        assert_eq!(b.sizes(), &[2, 6]); // #PYTHON list(b.shape)
        assert_eq!(
            b.f32s_ref()?,
            &[1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        ); // #PYTHON list(b.view(-1).tolist())
        Ok(())
    }

        #[test]
        fn test_flash_powder_ten_from_blob() -> StableTorchResult<()> {


            let d = Tensor::from(&[[1.0f32, 2.0], [3.0, 4.0]])?;

            let data = d.data()? ;
            let sizes = d.sizes();
            let strides = d.strides();
            let options = BlobOptionsBytes{
                sizes: sizes,
                strides: strides,
                dtype: d.dtype(),
                device: d.device(),
            };

            let ten_thing = Ten::from_bytes(data, &options)?;
            println!("ten_thing: {ten_thing:?}");

            Ok(())

        }
}
