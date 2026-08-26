//! Factory methods to create [`Tensor`].
//!
//! Pytorch puts these in the module, as torch.zeros(), I chose to put them as static methods on Tensor.

use crate::{StableTorchResult, Tensor, dtype::DType};
use torch_stable::aoti_torch::{AtenTensorHandle, aoti_torch_zero_};
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
                Self {
                    $v: Some(value),
                    ..Default::default()
                }
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
}
impl TensorFactory for Tensor {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_randn() -> StableTorchResult<()> {
        let d = Tensor::randn(&[1000, 1000], &Default::default())?;
        assert_eq!(d.sizes(), &[1000, 1000]);

        let mean = d.mean_dim(&Default::default())?;
        let value = mean.f32s_ref()?[0];
        assert!(value.abs() < 0.01);

        Ok(())
    }
}
