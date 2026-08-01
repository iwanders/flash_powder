//! This holds the core methods on the tensor object.
//!
//! Most of them originate from the yaml `native_functions.yaml` file.
//! See [native_functions.yaml@v2.12.0-rc7](https://github.com/pytorch/pytorch/blob/v2.12.0-rc7/aten/src/ATen/native/native_functions.yaml)
//! and its [readme](https://github.com/pytorch/pytorch/blob/v2.12.0-rc7/aten/src/ATen/native/README.md).
//!
//! Its readme states;
//! > Tensor operations as methods are appropriate for "core" Tensor operations (e.g., add, sub, etc.), but not for more complicated neural network layers (e.g., conv2d)
//!
//! This module holds the methods that are considered core tensor operations.

// https://docs.pytorch.org/docs/2.11/tensor_view.html
// has a nice overview of what operators return views.
//
// Hmm... from https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/README.md
// We should probably follow that guidance and kick conv2d to a functional module.
//
// The foo_ underscore methods modify data in place, see https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/README.md#annotations

use crate::dtype::DType;
use crate::factory::ToOptions;
use crate::properties::{TensorProperties, TensorPropertiesMut};
use crate::size::Size;
use crate::{StableTorchResult, Ten, TenMut, Tensor, TensorAccess, TensorAccessMut};
use torch_stable::{
    aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor, unsafe_call_bail,
    unsafe_call_dispatch_bail,
};
use torch_stable::{aoti_torch::*, unsafe_call_dispatch_panic};

use torch_stable::headeronly::core::MemoryFormat;
#[derive(Copy, Clone, Debug, Default)]
pub struct MeanOptions {
    pub dim: Option<usize>,
    pub keepdim: bool,
    pub dtype: Option<DType>,
}

#[derive(Copy, Clone, Debug)]
pub struct TopKOptions {
    pub dim: isize,
    pub largest: bool,
    pub sorted: bool,
}

impl Default for TopKOptions {
    fn default() -> Self {
        Self {
            dim: -1,
            largest: true,
            sorted: true,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum RoundingMode {
    /// Default behaviour, no rounding, if both operands are integers, the result is a scalar.
    #[default]
    Default,
    /// Rounds the division towards zero. Equivalent to C-style integer division.
    Truncate,
    /// Rounds the results of the division down. Equivalent to floor division in Python `//` and numpy's floor_divide
    Floor,
}

macro_rules! gen_compare_method {
    ($name:ident, $kernel_name:literal, $fancy_name:literal) => {
        #[doc = concat!( $fancy_name, "\n\nComparison with the ", stringify!($kernel_name), " kernel using self and other.")]
        fn $name<T: TensorAccess>(&self, other: &T) -> StableTorchResult<Tensor> {
            let mut stack: [StableIValue; 2] =
                [(self.get_tensor()).into(), other.get_tensor().into()];
            unsafe_call_dispatch_bail!($kernel_name, "Tensor", stack.as_mut_slice());
            let r: StableTensor = stack[0].try_into()?;
            Ok(Tensor::new(r))
        }
    };
}

/// Core methods that require const access.
///
/// See the [`core_methods`][crate::core_methods] module for description of this trait's functionality.
pub trait CoreMethods: TensorAccess + TensorProperties {
    /// Retrieve the shape as an owned [`Size`] object.
    fn shape(&self) -> Size {
        Size::from(self.sizes())
    }

    /// Narrow view
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.narrow.html)
    fn narrow<'a>(&'a self, dim: usize, start: isize, length: usize) -> StableTorchResult<Ten<'a>> {
        // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489
        let mut stack: [StableIValue; 4] = [
            self.get_tensor().into(),
            dim.into(),
            start.into(),
            length.into(),
        ];
        unsafe_call_dispatch_bail!("aten::narrow", "", stack.as_mut_slice());
        let marker = std::marker::PhantomData::<&'a ()>;
        Ok(Ten::new(marker, stack[0].try_into()?))
    }

    /// Conver the tensor to another tensor, returning an owning copy.
    ///
    /// The [`ToOptions`] struct can be created through `.into()` conversion, so the following works to create a tensor of U8s.
    ///
    /// ```rust
    /// # use flash_powder::prelude::*;
    /// # use flash_powder::{StableTorchResult, Tensor};
    /// # use flash_powder as fp;
    /// # fn foo(t: Tensor) -> StableTorchResult<()>{
    ///   let as_u8 = t.to(&fp::DType::U8.into())?;
    ///   let u8_on_cpu = as_u8.to(&fp::Device::CPU.into())?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L8033)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.to.html)
    ///
    fn to(&self, options: &ToOptions) -> StableTorchResult<Tensor> {
        const MAKE_COPY: bool = true;
        let mut stack: [StableIValue; 8] = [
            self.get_tensor().into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
            options.non_blocking.into(),
            MAKE_COPY.into(),
            (&options.memory_format).into(),
        ];
        unsafe_call_dispatch_bail!("aten::to", "dtype_layout", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_ne!(self.const_data_ptr(), r.const_data_ptr());

        Ok(Tensor::new(r))
    }

    /// View into a tensor
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L8362)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.view.html)
    ///
    fn view<'a>(&'a self, shape: &[usize]) -> StableTorchResult<Ten<'a>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), (shape).into()];
        unsafe_call_dispatch_bail!("aten::view", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        let marker = std::marker::PhantomData::<&'a ()>;
        Ok(Ten::new(marker, r))
    }

    /// Get a non mutable view of this tensor.
    fn ten<'a>(&'a self) -> StableTorchResult<Ten<'a>> {
        self.view(self.sizes())
    }

    /// Equal
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L10556)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.equal.html)
    ///
    fn is_equal<T: TensorAccess>(&self, other: &T) -> StableTorchResult<bool> {
        let mut stack: [StableIValue; 2] =
            [(self.get_tensor()).into(), (other.get_tensor()).into()];
        unsafe_call_dispatch_bail!("aten::equal", "", stack.as_mut_slice());
        let r: bool = stack[0].try_into()?;
        Ok(r)
    }

    /// Mean of this tensor.
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4055)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.mean.html)
    /// - [pytorch function](https://docs.pytorch.org/docs/2.12/generated/torch.mean.html#torch.mean)
    fn mean(&self, mean_options: &MeanOptions) -> StableTorchResult<Tensor> {
        // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489
        let as_array = mean_options.dim.as_ref().map(|z| [*z]);
        let as_array = as_array.as_ref().map(|a| a.as_slice());
        let mut stack: [StableIValue; 4] = [
            self.get_tensor().into(),
            (&as_array).into(),
            mean_options.keepdim.into(),
            (&mean_options.dtype).into(),
        ];
        unsafe_call_dispatch_bail!("aten::mean", "dim", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Min of this tensor.
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L9834)
    /// - [pytorch function](https://docs.pytorch.org/docs/2.13/generated/torch.min.html#torch.min)
    fn min(&self) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 1] = [self.get_tensor().into()];
        unsafe_call_dispatch_bail!("aten::min", "", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Max of this tensor.
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L9864)
    /// - [pytorch function](https://docs.pytorch.org/docs/2.13/generated/torch.max.html#torch.max)
    fn max(&self) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 1] = [self.get_tensor().into()];
        unsafe_call_dispatch_bail!("aten::max", "", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Perform a full clone of the tensor, not a lazy one.
    ///
    /// This is different from [`Tensor::clone`], which calls [`Self::lazy_clone`] because this actually performs the
    /// copy immediately. This is necessary in case we are copying from a Ten that does not own its data when it is
    /// instantiated through [`Ten::from_bytes`] .
    fn to_tensor(&self) -> StableTorchResult<Tensor> {
        let memory_format = MemoryFormat::Contiguous;
        let memory_format_opt = Some(memory_format);
        let mut stack: [StableIValue; 2] =
            [(self.get_tensor()).into(), (&memory_format_opt).into()];
        unsafe_call_dispatch_panic!("aten::clone", "", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Lazily clone this into an owning tensor.
    ///
    /// This only materializes the tensor if either the source or destination is written to.
    ///
    /// See also [`Self::to_tensor`] and [`Tensor::clone`]
    fn lazy_clone(&self) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 1] = [(self.get_tensor()).into()];
        unsafe_call_dispatch_panic!("aten::_lazy_clone", "", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Copied contigous version of tensor.
    ///
    /// Contrary to pytorch, this ALWAYS returns a copy.
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc9/aten/src/ATen/native/native_functions.yaml#L1715)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.contiguous.html)
    fn contiguous(&self) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 2] =
            [(self.get_tensor()).into(), MemoryFormat::Contiguous.into()];
        unsafe_call_dispatch_panic!("aten::contiguous", "", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r.clone())
    }

    /// Flatten the tensor
    ///
    /// Contrary to pytorch, this ALWAYS returns a copy.
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L2702)
    /// - [tensor method](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.flatten.html#torch.Tensor.flatten)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.11/generated/torch.flatten.html#torch.flatten)
    // Todo? maybe make this take a range? .flatten(..) or .flatten(3..) has a nice ring to it?
    fn flatten(&self, start_dim: usize, end_dim: Option<usize>) -> StableTorchResult<Tensor> {
        let end = end_dim.map(|z| z as isize).unwrap_or(-1);
        let mut stack: [StableIValue; 3] =
            [(self.get_tensor()).into(), start_dim.into(), end.into()];
        unsafe_call_dispatch_panic!("aten::flatten", "using_ints", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Division
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L2173)
    /// - [tensor method](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.div.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.11/generated/torch.div.html#torch.div)
    fn div<T: TensorAccess>(&self, other: &T) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), other.get_tensor().into()];
        unsafe_call_dispatch_panic!("aten::div", "Tensor", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Division with rounding mode
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L2106-L2112)
    /// - [tensor method](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.div.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.13/generated/torch.div.html#torch.div)
    fn div_mode<T: TensorAccess>(
        &self,
        other: &T,
        mode: RoundingMode,
    ) -> StableTorchResult<Tensor> {
        let string = match mode {
            RoundingMode::Default => None,
            RoundingMode::Truncate => Some("trunc"),
            RoundingMode::Floor => Some("floor"),
        };
        let mut stack: [StableIValue; 3] = [
            (self.get_tensor()).into(),
            other.get_tensor().into(),
            (&string).into(),
        ];
        unsafe_call_dispatch_panic!("aten::div", "Tensor_mode", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Modulus operation.
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L9814)
    /// - [tensor method](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.remainder.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.13/generated/torch.remainder.html)
    // Just like GE, this should also support scalars in the future.
    fn remainder<T: TensorAccess>(&self, other: &T) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), other.get_tensor().into()];
        unsafe_call_dispatch_panic!("aten::remainder", "Tensor", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Multiply
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4377)
    /// - [tensor method](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.mul.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.11/generated/torch.mul.html#torch.mul)
    fn mul<T: TensorAccess>(&self, other: &T) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), other.get_tensor().into()];
        unsafe_call_dispatch_panic!("aten::mul", "Tensor", stack.as_mut_slice());
        let r: Tensor = Tensor::new(stack[0].try_into().unwrap());
        Ok(r)
    }

    /// Add
    ///
    /// We can't actually dispatch into the kernel for addition yet, see [my comment](https://github.com/pytorch/pytorch/issues/174507#issuecomment-4150977835)
    /// about it and the reply on it.
    ///
    /// It would be the following kernel: [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L554)
    ///
    /// ~For now, we'll use the direct calls.~ The use of `aoti_torch_cpu_add_Tensor` and its CUDA flavour is problematic as that errors if we use integer tensors.
    /// So instead, we rely on `addcmul`, which conveniently has `_foreach` overload that takes the scalars in a Tensor itself.
    /// - [addcmul](https://docs.pytorch.org/docs/2.13/generated/torch.addcmul.html)
    /// - [native_functions](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L11241)
    // https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L11241
    fn add<T: TensorAccess + TensorProperties>(&self, other: &T) -> StableTorchResult<Tensor> {
        // Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars
        // Math is out_i = input_i + value * tensor1+_i + tensor2_i
        // For addition; value=1, tensor2_i = 1
        // According to https://docs.pytorch.org/docs/2.13/tensor_attributes.html scalars are least important in
        // promotion rules. So we can just make value and tensor2 always integers.
        let one: Tensor = 1u8.try_into()?;
        let self_array: &[StableIValue] = &[(self.get_tensor()).into()];
        let other_array: &[StableIValue] = &[(other.get_tensor()).into()];
        let tensor2_array: &[StableIValue] = &[(one.get_tensor()).into()];
        let scalars: Tensor = [1u8].try_into()?;
        let mut stack: [StableIValue; 4] = [
            self_array.into(),
            other_array.into(),
            tensor2_array.into(),
            scalars.get_tensor().into(),
        ];
        unsafe_call_dispatch_panic!("aten::_foreach_addcmul", "Tensor", stack.as_mut_slice());
        // This returns a list... How do we handle that?
        let v: Vec<StableIValue> = stack[0].try_into()?;
        let t: StableTensor = v
            .first()
            .copied()
            .ok_or(anyhow::format_err!("no value to retrieve"))?
            .try_into()?;
        let r: Tensor = Tensor::new(t);
        Ok(r)
    }

    /// Sub
    ///
    /// We can't actually dispatch into the kernel for addition yet, see [my comment](https://github.com/pytorch/pytorch/issues/174507#issuecomment-4150977835)
    /// about it and the reply on it.
    ///
    /// It would be the following kernel: [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L554)
    ///
    /// See [`Self::add`] for explanation on what we do now.
    fn sub<T: TensorAccess + TensorProperties>(&self, other: &T) -> StableTorchResult<Tensor> {
        // Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars
        // Math is out_i = input_i + value * tensor1+_i + tensor2_i
        // For addition; value=1, tensor2_i = 1
        // According to https://docs.pytorch.org/docs/2.13/tensor_attributes.html scalars are least important in
        // promotion rules. So we can just make value and tensor2 always integers.
        let one: Tensor = 1u8.try_into()?;
        let self_array: &[StableIValue] = &[(self.get_tensor()).into()];
        let other_array: &[StableIValue] = &[(other.get_tensor()).into()];
        let tensor2_array: &[StableIValue] = &[(one.get_tensor()).into()];
        let scalars: Tensor = [-1i8].try_into()?;
        let mut stack: [StableIValue; 4] = [
            self_array.into(),
            other_array.into(),
            tensor2_array.into(),
            scalars.get_tensor().into(),
        ];
        unsafe_call_dispatch_panic!("aten::_foreach_addcmul", "Tensor", stack.as_mut_slice());
        let v: Vec<StableIValue> = stack[0].try_into()?;
        let t: StableTensor = v
            .first()
            .copied()
            .ok_or(anyhow::format_err!("no value to retrieve"))?
            .try_into()?;
        let r: Tensor = Tensor::new(t);
        Ok(r)
    }

    /// Permute
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4675)
    /// - [tensor method](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.permute.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.11/generated/torch.permute.html#torch-permute)
    fn permute<'a>(&'a self, dims: &[usize]) -> StableTorchResult<Ten<'a>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), dims.into()];
        unsafe_call_dispatch_bail!("aten::permute", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());

        let marker = std::marker::PhantomData::<&'a ()>;
        Ok(Ten::new(marker, r))
    }

    /// Unsqueeze
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L6658)
    /// - [tensor method](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.unsqueeze.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.11/generated/torch.unsqueeze.html#torch.unsqueeze)
    fn unsqueeze<'a>(&'a self, dim: isize) -> StableTorchResult<Ten<'a>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), dim.into()];
        unsafe_call_dispatch_bail!("aten::unsqueeze", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        let marker = std::marker::PhantomData::<&'a ()>;
        Ok(Ten::new(marker, r))
    }

    /// Squeeze
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L5856)
    /// - [tensor method](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.squeeze.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.12/generated/torch.squeeze.html#torch.squeeze)
    fn squeeze(&self) -> StableTorchResult<Ten<'_>> {
        let mut stack: [StableIValue; 1] = [(self.get_tensor()).into()];
        unsafe_call_dispatch_bail!("aten::squeeze", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        let marker = std::marker::PhantomData::<&()>;
        Ok(Ten::new(marker, r))
    }
    /// Squeeze
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L5856)
    /// - [tensor method](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.squeeze.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.12/generated/torch.squeeze.html#torch.squeeze)
    fn squeeze_dim(&self, dim: isize) -> StableTorchResult<Ten<'_>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), dim.into()];
        unsafe_call_dispatch_bail!("aten::squeeze", "dim", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        let marker = std::marker::PhantomData::<&()>;
        Ok(Ten::new(marker, r))
    }
    /// Argmax
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L836)
    /// - [tensor method](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.argmax.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.12/generated/torch.argmax.html#torch.argmax)
    fn argmax(&self, dim: Option<isize>, keepdim: Option<bool>) -> StableTorchResult<Tensor> {
        let keepdim = keepdim.unwrap_or(false);
        let mut stack: [StableIValue; 3] =
            [(self.get_tensor()).into(), (&dim).into(), keepdim.into()];
        unsafe_call_dispatch_bail!("aten::argmax", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        Ok(Tensor::new(r))
    }

    // https://docs.pytorch.org/cppdocs/api/aten/indexing.html#tensor-indexing
    // the C++ API uses the index and index_put_ methods:

    /// Index with tensor.
    ///
    /// This retrieves a new tensor by looking up the indices.
    ///
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L3092-L3102)
    /// - pytorch method... I'm not actually sure :< it's just self\[indices\], but not sure to what that maps.
    fn index_tensor<T: TensorAccess>(&self, indices: &[T]) -> StableTorchResult<Tensor> {
        // func: index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
        let indices: Vec<Option<&StableTensor>> =
            indices.iter().map(|z| Some(z.get_tensor())).collect();
        let indices: Vec<StableIValue> = indices[..]
            .iter()
            .map(|z| {
                let a: StableIValue = z.into();
                a
            })
            .collect();
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), indices[..].into()];
        unsafe_call_dispatch_bail!("aten::index", "Tensor", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        Ok(Tensor::new(r))
    }

    /// Flip
    ///
    /// Reverse the order of an n-D tensor along given axis in dims.
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L6066-L6072)
    /// - [tensor method](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.flip.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.13/generated/torch.flip.html)
    fn flip(&self, dims: &[usize]) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), dims.into()];
        unsafe_call_dispatch_bail!("aten::flip", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        Ok(Tensor::new(r))
    }

    // Greater or Equal then
    //
    // - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L8879-L8885)
    // This function has like 5 overloads, the most important are Scalar and Tensor, for now we require TensorAccess
    // in the future, after Scalar is created, we can drop that req in Favour of a ScalarOrTensor trait, which would
    // allow us to handle both with the same function, and also support casting native types to Scalar.
    // fn ge<T: TensorAccess>(&self, other: &T) -> StableTorchResult<Tensor> {
    //     let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), other.get_tensor().into()];
    //     unsafe_call_dispatch_bail!("aten::ge", "Tensor", stack.as_mut_slice());
    //     let r: StableTensor = stack[0].try_into()?;
    //     Ok(Tensor::new(r))
    // }

    // - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L8879-L8885)
    gen_compare_method!(ne, "aten::ne", "Not Equal to");
    gen_compare_method!(eq, "aten::eq", "EQual to");
    gen_compare_method!(ge, "aten::ge", "Greater or Equal to");
    gen_compare_method!(gt, "aten::gt", "Greater Than");
    gen_compare_method!(le, "aten::le", "Less or Equal to");
    gen_compare_method!(lt, "aten::lt", "Less Than");

    /// Topk
    ///
    /// Returns the k largest elements of the given input tensor along a given dimension.
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L10012-L10017)
    /// - [tensor method](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.topk.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.13/generated/torch.topk.html#torch.topk)
    fn topk(&self, k: usize, options: &TopKOptions) -> StableTorchResult<(Tensor, Tensor)> {
        let mut stack: [StableIValue; 5] = [
            (self.get_tensor()).into(),
            k.into(),
            options.dim.into(),
            options.largest.into(),
            options.sorted.into(),
        ];
        unsafe_call_dispatch_bail!("aten::topk", "", stack.as_mut_slice());
        let values: StableTensor = stack[0].try_into()?;
        let indices: StableTensor = stack[1].try_into()?;
        Ok((Tensor::new(values), Tensor::new(indices)))
    }

    /// Move tensor to CPU
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L7231-L7232)
    /// - [tensor method](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.cpu.html)
    fn cpu(&self) -> StableTorchResult<Tensor> {
        let self_array: &[StableIValue] = &[(self.get_tensor()).into()];
        let mut stack: [StableIValue; 1] = [self_array.into()];
        unsafe_call_dispatch_panic!("aten::_to_cpu", "", stack.as_mut_slice());
        let v: Vec<StableIValue> = stack[0].try_into()?;
        let t: StableTensor = v
            .first()
            .copied()
            .ok_or(anyhow::format_err!("no value to retrieve"))?
            .try_into()?;
        // That function returns the same value if its already on the CPU, we MUST return a COW flavour.
        let r: Tensor = Tensor::new(t).lazy_clone()?;
        Ok(r)
    }
}
impl CoreMethods for Tensor {}
impl<'a> CoreMethods for Ten<'a> {}
impl<'a> CoreMethods for TenMut<'a> {}

impl<'a> Ten<'a> {
    /// Narrow view
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.narrow.html)
    pub fn narrow(&self, dim: usize, start: isize, length: usize) -> StableTorchResult<Ten<'a>> {
        // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489

        let mut stack: [StableIValue; 4] = [
            self.get_tensor().into(),
            dim.into(),
            start.into(),
            length.into(),
        ];
        unsafe_call_dispatch_bail!("aten::narrow", "", stack.as_mut_slice());
        Ok(Ten::new(self.as_parent(), stack[0].try_into()?))
    }

    pub fn select(&self, dim: usize, index: usize) -> StableTorchResult<Ten<'a>> {
        let mut stack: [StableIValue; 3] = [self.get_tensor().into(), dim.into(), index.into()];
        unsafe_call_dispatch_bail!("aten::select", "int", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(Ten::new(self.as_parent(), r))
    }

    pub fn squeeze(&self) -> StableTorchResult<Ten<'a>> {
        let mut stack: [StableIValue; 1] = [(self.get_tensor()).into()];
        unsafe_call_dispatch_bail!("aten::squeeze", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        Ok(Ten::new(self.as_parent(), r))
    }
    /// Unsqueeze
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L6658)
    /// - [tensor method](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.unsqueeze.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.11/generated/torch.unsqueeze.html#torch.unsqueeze)
    pub fn unsqueeze(&self, dim: isize) -> StableTorchResult<Ten<'a>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), dim.into()];
        unsafe_call_dispatch_bail!("aten::unsqueeze", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        Ok(Ten::new(self.as_parent(), r))
    }
}

/// Core methods that require mutable access.
///
/// See the [`core_methods`][crate::core_methods] module for description of this trait's functionality.
pub trait CoreMethodsMut: TensorAccessMut + TensorPropertiesMut {
    fn narrow_mut(
        &mut self,
        dim: usize,
        start: usize,
        end: usize,
    ) -> StableTorchResult<TenMut<'_>> {
        let mut stack: [StableIValue; 4] = [
            self.get_tensor().into(),
            dim.into(),
            start.into(),
            end.into(),
        ];
        // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489
        unsafe_call_dispatch_bail!("aten::narrow", "", stack.as_mut_slice());

        Ok(TenMut::new(self.get_tensor_mut(), stack[0].try_into()?))
    }

    /// Fill a tensor with another tensor.
    ///
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L2730)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.fill_.html)
    fn fill_tensor<T: TensorAccess>(&mut self, value: &T) -> StableTorchResult<()> {
        let mut stack: [StableIValue; 2] =
            [(self.get_tensor()).into(), (value.get_tensor()).into()];
        unsafe_call_dispatch_bail!("aten::fill_", "Tensor", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        let retrieve = Tensor::new(r);
        assert_eq!(retrieve.const_data_ptr(), self.const_data_ptr());
        Ok(())
    }
    fn fill_f64(&mut self, value: f64) -> StableTorchResult<()> {
        unsafe_call_bail!(aoti_torch_aten_fill__Scalar(self.get_tensor().get(), value));
        Ok(())
    }

    /// View into a tensor
    ///
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L8362)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.view.html)
    ///
    fn view_mut(&mut self, shape: &[usize]) -> StableTorchResult<TenMut<'_>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), (shape).into()];
        unsafe_call_dispatch_bail!("aten::view", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        Ok(TenMut::new(self.get_tensor_mut(), r))
    }

    fn ten_mut<'a>(&'a mut self) -> StableTorchResult<TenMut<'a>> {
        let shape = self.sizes();
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), (shape).into()];
        unsafe_call_dispatch_bail!("aten::view", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        Ok(TenMut::new(self.get_tensor_mut(), r))
    }

    /// Permute
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4675)
    /// - [tensor method](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.permute.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.11/generated/torch.permute.html#torch-permute)
    fn permute_mut(&mut self, dims: &[usize]) -> StableTorchResult<TenMut<'_>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), dims.into()];
        unsafe_call_dispatch_bail!("aten::permute", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        Ok(TenMut::new(self.get_tensor_mut(), r))
    }

    /// Assign (.copy_); copies the elements from src into self tensor and returns self.
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L1808)
    /// - [tensor method](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.copy_.html)
    fn copy_from_tensor<T: TensorAccess>(&mut self, t: &T) -> StableTorchResult<()> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), t.get_tensor().into()];
        unsafe_call_dispatch_bail!("aten::copy_", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.const_data_ptr(), r.const_data_ptr());
        Ok(())
    }
}
impl CoreMethodsMut for Tensor {}
// impl<'a> CoreMethodsMut for Ten<'a> {}
impl<'a> CoreMethodsMut for TenMut<'a> {}

impl<'a> TenMut<'a> {
    /// Narrow mut view
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.narrow.html)
    pub fn into_narrow_mut(
        self,
        dim: usize,
        start: isize,
        length: usize,
    ) -> StableTorchResult<TenMut<'a>> {
        // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489

        let mut stack: [StableIValue; 4] = [
            self.get_tensor().into(),
            dim.into(),
            start.into(),
            length.into(),
        ];
        unsafe_call_dispatch_bail!("aten::narrow", "", stack.as_mut_slice());
        Ok(TenMut::new(self.into_parent(), stack[0].try_into()?))
    }

    pub fn into_select_mut(self, dim: usize, index: usize) -> StableTorchResult<TenMut<'a>> {
        let mut stack: [StableIValue; 3] = [self.get_tensor().into(), dim.into(), index.into()];
        unsafe_call_dispatch_bail!("aten::select", "int", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(TenMut::new(self.into_parent(), r))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{index::TensorIndex as _, prelude::*};

    #[test]
    fn test_flash_powder_fill() -> StableTorchResult<()> {
        /*
            #|PYTHON
            t = torch.zeros([2,2])
            t.fill_(3.0)
            v = torch.tensor(5.0)
        */

        let mut t = Tensor::zeros(&[2, 2], &Default::default())?;
        assert_eq!(t.sizes(), &[2, 2]); // #PYTHON list(t.shape)

        let v = Tensor::from_f32(5.0)?;
        assert_eq!(v.f32s_ref()?, &[5.0f32]); // #PYTHON list(v.view(-1).tolist())

        /*
            #|PYTHON
            t.fill_(v)
        */

        t.fill_tensor(&v)?;
        assert_eq!(t.f32s_ref()?, &[5.0f32, 5.0, 5.0, 5.0]); // #PYTHON list(t.view(-1).tolist())

        Ok(())
    }

    #[test]
    fn test_flash_powder_narrow() -> StableTorchResult<()> {
        /*
            #|PYTHON
            t = torch.tensor(list(range(1,10)), dtype=torch.float).reshape([3,3])
            v = t.narrow(0, 0, 3)
            v.fill_(3.0)
            nv = t.narrow(0, 0, 3)
        */

        let mut t = Tensor::zeros(&[3, 3], &Default::default())?;
        assert_eq!(t.sizes(), &[3, 3]); // #PYTHON list(t.shape)

        let mut view_mut = t.narrow_mut(0, 0, 3)?;
        view_mut.fill_tensor(&Tensor::from_f32(3.0)?)?;
        assert_eq!(view_mut.sizes(), &[3, 3]); // #PYTHON list(v.shape)
        assert_eq!(
            view_mut.f32s_ref()?,
            &[3.0f32, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        ); // #PYTHON list(v.view(-1).tolist())

        drop(view_mut);

        let view = t.narrow(0, 0, 3)?;
        assert_eq!(
            view.f32s_ref()?,
            &[3.0f32, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        ); // #PYTHON list(nv.view(-1).tolist())

        // from https://docs.pytorch.org/docs/2.11/generated/torch.narrow.html#torch.narrow
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,10)), dtype=torch.float).reshape([3,3])
        */

        let d = Tensor::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])?;
        assert_eq!(d.sizes(), &[3, 3]); // #PYTHON list(d.shape)
        assert_eq!(
            d.f32s_ref()?,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ); // #PYTHON list(d.view(-1).tolist())

        /*
            #|PYTHON
            x = torch.narrow(d, 0, 0, 2)
        */
        let x = d.narrow(0, 0, 2)?;
        assert_eq!(x.sizes(), &[2, 3]); // #PYTHON list(x.shape)

        assert_eq!(x.f32_ref(&[0, 0])?, &1.0); // #PYTHON x[ 0,  0].item()
        assert_eq!(x.f32_ref(&[0, 1])?, &2.0); // #PYTHON x[ 0,  1].item()
        assert_eq!(x.f32_ref(&[0, 2])?, &3.0); // #PYTHON x[ 0,  2].item()
        assert_eq!(x.f32_ref(&[1, 0])?, &4.0); // #PYTHON x[ 1,  0].item()
        assert_eq!(x.f32_ref(&[1, 1])?, &5.0); // #PYTHON x[ 1,  1].item()
        assert_eq!(x.f32_ref(&[1, 2])?, &6.0); // #PYTHON x[ 1,  2].item()

        /*
            #|PYTHON
            x = torch.narrow(d, 1, 1, 2)
        */
        let x = d.narrow(1, 1, 2)?;
        assert_eq!(x.sizes(), &[3, 2]); // #PYTHON list(x.shape)
        assert_eq!(x.is_contiguous(), false);
        assert_eq!(x.i((0, 0))?.as_f32()?, &2.0); // #PYTHON x[ 0, 0].item()
        assert_eq!(x.i((1, 0))?.as_f32()?, &5.0); // #PYTHON x[ 1, 0].item()
        assert_eq!(x.i((2, 0))?.as_f32()?, &8.0); // #PYTHON x[ 2, 0].item()
        assert_eq!(x.i((0, 1))?.as_f32()?, &3.0); // #PYTHON x[ 0, 1].item()
        assert_eq!(x.i((1, 1))?.as_f32()?, &6.0); // #PYTHON x[ 1, 1].item()
        assert_eq!(x.i((2, 1))?.as_f32()?, &9.0); // #PYTHON x[ 2, 1].item()

        /*
            #|PYTHON
            x = torch.narrow(d, 1, -3, 2)
        */
        let x = d.narrow(1, -3, 2)?;
        assert_eq!(x.sizes(), &[3, 2]); // #PYTHON list(x.shape)
        assert_eq!(x.is_contiguous(), false);
        assert_eq!(x.i((0, 0))?.as_f32()?, &1.0); // #PYTHON x[ 0, 0].item()
        assert_eq!(x.i((1, 0))?.as_f32()?, &4.0); // #PYTHON x[ 1, 0].item()
        assert_eq!(x.i((2, 0))?.as_f32()?, &7.0); // #PYTHON x[ 2, 0].item()
        assert_eq!(x.i((0, 1))?.as_f32()?, &2.0); // #PYTHON x[ 0, 1].item()
        assert_eq!(x.i((1, 1))?.as_f32()?, &5.0); // #PYTHON x[ 1, 1].item()
        assert_eq!(x.i((2, 1))?.as_f32()?, &8.0); // #PYTHON x[ 2, 1].item()

        /*
            #|PYTHON
            d = d.detach().clone()
            x = torch.narrow(d, 0, 0, 2)
            x[0,0] = 15.0
            x[0,2] = 16.0
            x[1,2] = 17.0
        */
        let mut d: Tensor = d.clone();
        let mut x = d.narrow_mut(0, 0, 2)?;
        assert_eq!(x.sizes(), &[2, 3]); // #PYTHON list(x.shape)

        *x.f32_mut(&[0, 0])? = 15.0;
        *x.f32_mut(&[0, 2])? = 16.0;
        *x.f32_mut(&[1, 2])? = 17.0;

        assert_eq!(d.f32_ref(&[0, 0])?, &15.0); // #PYTHON d[ 0,  0].item()
        assert_eq!(d.f32_ref(&[0, 2])?, &16.0); // #PYTHON d[ 0,  2].item()
        assert_eq!(d.f32_ref(&[1, 2])?, &17.0); // #PYTHON d[ 1,  2].item()

        Ok(())
    }
    #[test]
    fn test_flash_powder_aten_empty() -> StableTorchResult<()> {
        let _ = Tensor::empty(&[5, 5], &Default::default())?.fill_f64(0.0);
        Ok(())
    }

    #[test]
    fn test_flash_powder_to() -> StableTorchResult<()> {
        use crate::factory::TensorOptions;
        let t = Tensor::zeros(
            &[5, 5],
            &TensorOptions {
                ..Default::default()
            },
        )?;
        assert_eq!(t.dtype(), DType::F32);
        let orig = t.const_data_ptr();

        let z = t.to(&ToOptions {
            ..Default::default()
        })?;
        assert_eq!(z.storage_offset(), 0);
        assert_ne!(orig, z.const_data_ptr());

        Ok(())
    }

    #[test]
    fn test_flash_powder_view() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([4,4])
        */
        let mut d = Tensor::zeros(&[16], &Default::default())?;
        for (i, v) in d.f32s_mut()?.iter_mut().enumerate() {
            *v = (i + 1) as f32
        }

        let mut a = d.view_mut(&[4, 4])?;

        assert_eq!(a.sizes(), &[4, 4]); // #PYTHON list(d.shape)
        assert_eq!(
            a.f32s_ref()?,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        );
        a.f32s_mut()?[0] = 50.0;

        assert_eq!(a.f32s_mut()?[0], 50.0);
        assert_eq!(a.f32s_ref()?[0], 50.0);

        let mut n = a.lazy_clone()?;
        // Currently lazy copy
        let old_n_ptr = n.const_data_ptr();
        assert_eq!(n.const_data_ptr(), a.const_data_ptr());
        assert!(n.is_equal(&a)?);

        // Verify n holds same data
        assert_eq!(n.f32s_ref()?[0], 50.0);
        // Modify n, this performs the copy.
        n.f32s_mut()?[0] = 20.0;
        assert_eq!(n.is_equal(&a)?, false);

        // data pointer shouldn't be the same now.
        assert_ne!(n.const_data_ptr(), old_n_ptr);

        assert_eq!(d.f32s_mut()?[0], 50.0);

        // Try a non owning view
        let v = d.view(&[16])?;
        let mut cv = v.lazy_clone()?;
        cv.f32s_mut()?[0] = 10.0;
        assert_eq!(cv.f32s_ref()?[0], 10.0);
        assert_eq!(v.f32s_ref()?[0], 50.0);

        // Reshape to incorrect size.
        assert!(d.view(&[12]).is_err());

        Ok(())
    }

    #[test]
    fn test_flash_powder_mean() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([1,4,4])
            mean = d.mean()
            mean_0 = d.mean(0)
            mean_1 = d.mean(1)
            mean_2 = d.mean(2)
            mean_1_double = d.mean(1, dtype=torch.double)
        */

        let d = Tensor::from([[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]])?;
        assert_eq!(d.sizes(), &[1, 4, 4]); // #PYTHON list(d.shape)

        let mean = d.mean(&Default::default())?;
        assert_eq!(mean.dim(), 0); // #PYTHON mean.dim()
        assert_eq!(mean.f32s_ref()?, &[8.5f32]); // #PYTHON list(mean.view(-1).tolist())

        let mean_0 = d.mean(&MeanOptions {
            dim: Some(0),
            ..Default::default()
        })?;
        assert_eq!(mean_0.sizes(), &[4, 4]); // #PYTHON list(mean_0.shape)
        assert_eq!(
            mean_0.f32s_ref()?,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(mean_0.view(-1).tolist())

        let mean_1 = d.mean(&MeanOptions {
            dim: Some(1),
            ..Default::default()
        })?;
        assert_eq!(mean_1.sizes(), &[1, 4]); // #PYTHON list(mean_1.shape)
        assert_eq!(mean_1.f32s_ref()?, &[7.0f32, 8.0, 9.0, 10.0]); // #PYTHON list(mean_1.view(-1).tolist())

        let mean_2 = d.mean(&MeanOptions {
            dim: Some(2),
            ..Default::default()
        })?;
        assert_eq!(mean_2.sizes(), &[1, 4]); // #PYTHON list(mean_2.shape)
        assert_eq!(mean_2.f32s_ref()?, &[2.5f32, 6.5, 10.5, 14.5]); // #PYTHON list(mean_2.view(-1).tolist())

        let mean_1_double = d.mean(&MeanOptions {
            dim: Some(1),
            dtype: Some(DType::F64),
            ..Default::default()
        })?;
        assert_eq!(mean_1_double.sizes(), &[1, 4]); // #PYTHON list(mean_1_double.shape)
        assert_eq!(mean_1_double.f64s_ref()?, &[7.0f64, 8.0, 9.0, 10.0]); // #PYTHON list(mean_1_double.view(-1).tolist())
        Ok(())
    }

    #[test]
    fn test_flash_powder_full_view() -> StableTorchResult<()> {
        let d = Tensor::from([[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]])?;
        let z = d.view(d.sizes())?;
        let z = z.ten()?;

        drop(z);

        let mut d = d;
        let mut z = d.ten_mut()?;
        let mut z = z.ten_mut()?;
        *z.f32_mut(&[0, 0])? = 30.0;
        assert_eq!(d.f32_ref(&[0, 0])?, &30.0);

        let shape = d.shape();
        println!("shape: {shape:?}");
        let z = d.view_mut(&shape)?;
        assert_eq!(z.f32_ref(&[0, 0])?, &30.0);

        Ok(())
    }

    #[test]
    fn test_flash_powder_contiguous() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([1,4,4])
        */

        let d = Tensor::from([[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]])?;
        assert_eq!(d.sizes(), &[1, 4, 4]); // #PYTHON list(d.shape)

        /*
            #|PYTHON
            v = d[0, 0:2, 0:3]
        */
        let v = d.i((0, 0..2, 0..3))?;
        assert_eq!(v.sizes(), &[2, 3]); // #PYTHON list(v.shape)
        assert_eq!(v.is_contiguous(), false);

        let v_c = v.contiguous()?;
        assert_eq!(v_c.is_contiguous(), true);
        assert_eq!(v.is_equal(&v_c)?, true);

        Ok(())
    }
    #[test]
    fn test_flash_powder_flatten() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.11/generated/torch.flatten.html#torch.flatten
        /*
            #|PYTHON
            t = torch.tensor([[[1, 2],
                               [3, 4]],
                              [[5, 6],
                               [7, 8]]])
        */

        let t = Tensor::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])?;
        assert_eq!(t.sizes(), &[2, 2, 2]); // #PYTHON list(t.shape)

        /*
            #|PYTHON
            l = torch.flatten(t)
            two = torch.flatten(t, 1)
        */
        let l = t.flatten(0, None)?;
        assert_eq!(l.sizes(), &[8]); // #PYTHON list(l.shape)

        let l = t.flatten(1, None)?;
        assert_eq!(l.sizes(), &[2, 4]); // #PYTHON list(two.shape)

        Ok(())
    }

    #[test]
    fn test_flash_powder_div() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.11/generated/torch.div.html#torch.div
        /*
            #|PYTHON
            x = torch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
            r = torch.div(x, 0.5)
        */

        let t = Tensor::from([0.3810f32, 1.2774, -0.2972, -0.3719, 0.4637])?;
        let denom: Tensor = 0.5.try_into()?;
        let r = t.div(&denom)?;
        assert_eq!(r.sizes(), &[5]); // #PYTHON list(r.shape)
        assert_eq!(
            r.f32s_ref()?,
            &[
                0.7620000243186951f32,
                2.554800033569336,
                -0.5943999886512756,
                -0.7437999844551086,
                0.9273999929428101
            ]
        ); // #PYTHON list(r.view(-1).tolist())

        Ok(())
    }

    #[test]
    fn test_flash_powder_div_mode() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.11/generated/torch.div.html#torch.div
        // Simplifying this example to just two rows, otherwise the amount of numbers gets soo large, but same numbers
        // from the docs.
        /*
            #|PYTHON
            a = torch.tensor([-0.3711, -1.9353, -0.4605, -0.2917])
            b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
            a_div_b = torch.div(a, b)
            a_trunc_b = torch.div(a, b, rounding_mode="trunc")
            a_floor_b = torch.div(a, b, rounding_mode="floor")
        */

        let a: Tensor = [-0.3711f32, -1.9353, -0.4605, -0.2917].try_into()?;
        let b: Tensor = [0.8032f32, 0.2930, -0.8113, -0.2308].try_into()?;

        let a_div_b = a.div_mode(&b, RoundingMode::Default)?;
        assert_eq!(a_div_b.sizes(), &[4]); // #PYTHON list(a_div_b.shape)
        assert_eq!(
            a_div_b.f32s_ref()?,
            &[
                -0.4620268940925598f32,
                -6.605119228363037,
                0.567607581615448,
                1.2638648748397827
            ]
        ); // #PYTHON list(a_div_b.view(-1).tolist())

        let a_trunc_b = a.div_mode(&b, RoundingMode::Truncate)?;
        assert_eq!(a_trunc_b.sizes(), &[4]); // #PYTHON list(a_trunc_b.shape)
        assert_eq!(a_trunc_b.f32s_ref()?, &[-0.0f32, -6.0, 0.0, 1.0]); // #PYTHON list(a_trunc_b.view(-1).tolist())

        let a_floor_b = a.div_mode(&b, RoundingMode::Floor)?;
        assert_eq!(a_floor_b.sizes(), &[4]); // #PYTHON list(a_floor_b.shape)
        assert_eq!(a_floor_b.f32s_ref()?, &[-1.0f32, -7.0, 0.0, 1.0]); // #PYTHON list(a_floor_b.view(-1).tolist())
        Ok(())
    }

    #[test]
    fn test_flash_powder_mul() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.11/generated/torch.mul.html#torch.mul
        /*
            #|PYTHON
            x = torch.tensor([ 0.2015, -0.4255,  2.6087])
            r = torch.mul(x, 100.0)
        */

        let t = Tensor::from([0.2015f32, -0.4255, 2.6087])?;
        let factor: Tensor = 100.0.try_into()?;
        let r = t.mul(&factor)?;
        assert_eq!(r.sizes(), &[3]); // #PYTHON list(r.shape)
        assert_eq!(
            r.f32s_ref()?,
            &[20.149999618530273f32, -42.54999923706055, 260.8699951171875]
        ); // #PYTHON list(r.view(-1).tolist())

        Ok(())
    }
    #[test]
    fn test_flash_powder_permute() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.11/generated/torch.mul.html#torch.mul
        /*
            #|PYTHON
            x = torch.randn(2, 3, 5)
            y = x.permute(2, 0, 1)
        */

        let mut x = Tensor::randn(&[2, 3, 5], &Default::default())?;
        assert_eq!(x.sizes(), &[2, 3, 5]); // #PYTHON list(x.shape)
        let y = x.permute(&[2, 0, 1])?;
        assert_eq!(y.sizes(), &[5, 2, 3]); // #PYTHON list(y.shape)

        let z = x.permute_mut(&[2, 0, 1])?;
        assert_eq!(z.is_contiguous(), false);
        // println!("z: { :?}", z.shape());

        // *z.f32_mut(&[3, 1, 2])? = 3.30;
        // assert_eq!(x.f32_ref(&[2, 3, 1])?, &3.30);

        Ok(())
    }
    #[test]
    fn test_flash_powder_unsqueeze() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.11/generated/torch.unsqueeze.html#torch.unsqueeze
        /*
            #|PYTHON
            x = torch.tensor([1,2,3,4])
            y1 = torch.unsqueeze(x, 0)
            y2 = torch.unsqueeze(x, 1)
        */

        let x: Tensor = [1, 2, 3, 4].try_into()?;
        assert_eq!(x.sizes(), &[4]); // #PYTHON list(x.shape)
        let y1 = x.unsqueeze(0)?;
        assert_eq!(y1.sizes(), &[1, 4]); // #PYTHON list(y1.shape)
        let y2 = x.unsqueeze(1)?;
        assert_eq!(y2.sizes(), &[4, 1]); // #PYTHON list(y2.shape)

        Ok(())
    }
    #[test]
    fn test_flash_powder_argmax() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.12/generated/torch.argmax.html#torch.argmax
        /*
            #|PYTHON
            a = torch.tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [ 0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092,  0.5419, -0.2993,  0.3195]])
            y1 = torch.argmax(a)
            y2 = torch.argmax(a, dim=1)
        */

        let x: Tensor = [
            [1.3398, 0.2663, -0.2686, 0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092, 0.5419, -0.2993, 0.3195],
        ]
        .try_into()?;
        assert_eq!(x.sizes(), &[4, 4]); // #PYTHON list(a.shape)
        let y1 = x.argmax(None, None)?;
        assert_eq!(y1.sizes(), &[]); // #PYTHON list(y1.shape)
        assert_eq!(y1.as_i64()?, &0); // #PYTHON y1.item()
        let y2 = x.argmax(Some(1), None)?;
        assert_eq!(y2.sizes(), &[4]); // #PYTHON list(y2.shape)
        assert_eq!(y2.i64s_ref()?, &[0, 2, 0, 1]); // #PYTHON y2.tolist()

        Ok(())
    }

    #[test]
    fn test_flash_powder_squeeze() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.12/generated/torch.squeeze.html#torch.squeeze
        /*
            #|PYTHON
            x = torch.zeros(2, 1, 2, 1, 2)
            y1 = torch.squeeze(x)
            y2 = torch.squeeze(x, 0)
            y3 = torch.squeeze(x, 1)
            #y3 = torch.squeeze(x, (1,2,3))
        */

        let x: Tensor = Tensor::zeros(&[2, 1, 2, 1, 2], &Default::default())?;
        assert_eq!(x.sizes(), &[2, 1, 2, 1, 2]); // #PYTHON list(x.shape)
        let y1 = x.squeeze()?;
        assert_eq!(y1.sizes(), &[2, 2, 2]); // #PYTHON list(y1.shape)

        let y2 = x.squeeze_dim(0)?;
        assert_eq!(y2.sizes(), &[2, 1, 2, 1, 2]); // #PYTHON list(y2.shape)

        let y3 = x.squeeze_dim(1)?;
        assert_eq!(y3.sizes(), &[2, 2, 1, 2]); // #PYTHON list(y3.shape)

        Ok(())
    }
    #[test]
    fn test_flash_powder_copy_from_tensor() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.12/generated/torch.squeeze.html#torch.squeeze
        /*
            #|PYTHON
            x = torch.tensor([1.0, 2.0, 3.0])
            x2 = torch.tensor([0.0, 1.0, 0.0])
        */

        let mut x: Tensor = [1.0, 2.0, 3.0].try_into()?;
        assert_eq!(x.f64s_ref()?, &[1.0, 2.0, 3.0]); // #PYTHON x.tolist()
        let x2: Tensor = [0.0, 1.0, 0.0].try_into()?;
        assert_eq!(x2.f64s_ref()?, &[0.0, 1.0, 0.0]); // #PYTHON x2.tolist()

        /*
            #|PYTHON
            x.copy_(x2)
        */
        x.copy_from_tensor(&x2)?;
        assert_eq!(x.f64s_ref()?, &[0.0, 1.0, 0.0]); // #PYTHON x.tolist()

        // Can this assign into a view?
        /*
            #|PYTHON
            x = torch.tensor([1.0, 2.0, 3.0])
            x2 = torch.tensor([0.0, 1.0, 0.0])
            x[0:2].copy_(x2[0:2])
        */
        let x2: Tensor = [0.0, 1.0, 0.0].try_into()?;
        let mut x: Tensor = [1.0, 2.0, 3.0].try_into()?;
        x.i_mut(0..2)?.copy_from_tensor(&x2.i(0..2)?)?;
        assert_eq!(x.f64s_ref()?, &[0.0, 1.0, 3.0]); // #PYTHON x.tolist()

        Ok(())
    }

    #[test]
    fn test_flash_powder_lazy_clone_to_owned() -> StableTorchResult<()> {
        let x: Tensor = [1.0, 2.0, 3.0].try_into()?;
        let x_clone = x.clone();
        assert_eq!(x.const_data_ptr(), x_clone.const_data_ptr());
        let x_lazy = x.lazy_clone()?;
        assert_eq!(x.const_data_ptr(), x_lazy.const_data_ptr());
        // And an owning clone.
        let x_owned = x.to_tensor()?;
        assert_ne!(x.const_data_ptr(), x_owned.const_data_ptr());

        Ok(())
    }

    #[test]
    fn test_flash_powder_core_method_index_tensor() -> StableTorchResult<()> {
        /*
            #|PYTHON
            color_lookup = torch.tensor([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])

            the_indices = torch.tensor([0, 1, 2, 2, 1, 0, 1,2,0], dtype=torch.long)

            combined = color_lookup[the_indices]
        */
        let color_lookup = Tensor::from([[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])?;
        assert_eq!(color_lookup.sizes(), &[3, 3]); // #PYTHON list(color_lookup.shape)
        assert_eq!(color_lookup.dtype(), DType::F32); // #PYTHON color_lookup.dtype

        let the_indices = Tensor::from([0i64, 1, 2, 2, 1, 0, 1, 2, 0])?;
        assert_eq!(the_indices.sizes(), &[9]); // #PYTHON list(the_indices.shape)
        assert_eq!(the_indices.dtype(), DType::I64); // #PYTHON the_indices.dtype

        let combined = color_lookup.index_tensor(&[the_indices])?;
        assert_eq!(combined.sizes(), &[9, 3]); // #PYTHON list(combined.shape)

        assert_eq!(
            combined.f32s_ref()?,
            &[
                1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
            ]
        ); // #PYTHON combined.ravel().tolist()

        Ok(())
    }

    #[test]
    fn test_flash_powder_addition_subtract() -> StableTorchResult<()> {
        /*
            #|PYTHON
            x = torch.tensor([1.0, 2.0, 3.0])
            x2 = torch.tensor([0.0, 1.0, 6.0])
            added = x + x2
            sub = x - x2
        */
        let x: Tensor = [1.0, 2.0, 3.0].try_into()?;
        assert_eq!(x.f64s_ref()?, &[1.0, 2.0, 3.0]); // #PYTHON x.tolist()
        let x2: Tensor = [0.0, 1.0, 6.0].try_into()?;
        assert_eq!(x2.f64s_ref()?, &[0.0, 1.0, 6.0]); // #PYTHON x2.tolist()

        let added = x.add(&x2)?;
        assert_eq!(added.f64s_ref()?, &[1.0f64, 3.0, 9.0]); // #PYTHON added.ravel().tolist()

        let sub = x.sub(&x2)?;
        assert_eq!(sub.f64s_ref()?, &[1.0f64, 1.0, -3.0]); // #PYTHON sub.ravel().tolist()

        // And for integer, because that's broken :(
        /*
            #|PYTHON
            x = torch.tensor([1, 2, 3])
            x2 = torch.tensor([0, 1, 6])
            added = x + x2
            sub = x - x2
        */
        let x: Tensor = [1, 2, 3].try_into()?;
        let x2: Tensor = [0, 1, 6].try_into()?;
        let added = x.add(&x2)?;
        assert_eq!(added.i32s_ref()?, &[1i32, 3, 9]); // #PYTHON added.ravel().tolist()

        let sub = x.sub(&x2)?;
        assert_eq!(sub.i32s_ref()?, &[1i32, 1, -3]); // #PYTHON sub.ravel().tolist()

        // Test i8

        /*
            #|PYTHON
            x = torch.tensor([1, 2, 3], dtype=torch.int8)
            x2 = torch.tensor([0, 5, 0], dtype=torch.int8)
            added = x + x2
            sub = x - x2
        */
        let x: Tensor = [1i8, 2, 3].try_into()?;
        let x2: Tensor = [0i8, 5, 0].try_into()?;
        let added = x.add(&x2)?;
        assert_eq!(added.i8s_ref()?, &[1i8, 7, 3]); // #PYTHON added.ravel().tolist()

        let sub = x.sub(&x2)?;
        assert_eq!(sub.i8s_ref()?, &[1i8, -3, 3]); // #PYTHON sub.ravel().tolist()

        // Also verify that we can do larger numbers and they don't get rounded on u8 or something silly.
        /*
            #|PYTHON
            x = torch.tensor([1<<50, 1<<62, 0], dtype=torch.int64)
            x2 = torch.tensor([1337, 1337, 1337], dtype=torch.int64)
            added = x + x2
            sub = x - x2
        */
        let x: Tensor = [(1i64 << 50), (1 << 62), 0].try_into()?;
        let x2: Tensor = [1337i64, 1337, 1337].try_into()?;

        let added = x.add(&x2)?;
        assert_eq!(
            added.i64s_ref()?,
            &[1125899906843961i64, 4611686018427389241, 1337]
        ); // #PYTHON added.ravel().tolist()

        let sub = x.sub(&x2)?;
        assert_eq!(
            sub.i64s_ref()?,
            &[1125899906841287i64, 4611686018427386567, -1337]
        ); // #PYTHON sub.ravel().tolist()

        // And also check some larger floats
        /*
            #|PYTHON
            x = torch.tensor([1.0e8, 2.0e16, 3.0e30], dtype=torch.double)
            x2 = torch.tensor([1.0e9, 2e18, 0.0], dtype=torch.double)
            added = x + x2
            sub = x - x2
        */
        let x: Tensor = [1.0e8, 2.0e16, 3.0e30].try_into()?;
        assert_eq!(x.f64s_ref()?, &[100000000.0, 2e+16, 3e+30]); // #PYTHON x.tolist()
        let x2: Tensor = [1.0e9, 2e18, 0.0].try_into()?;
        assert_eq!(x2.f64s_ref()?, &[1000000000.0, 2e+18, 0.0]); // #PYTHON x2.tolist()
        Ok(())
    }

    #[test]
    fn test_flash_powder_min_and_max() -> StableTorchResult<()> {
        /*
            #|PYTHON
            x = torch.tensor([1.0, 2.0, 3.0])
            min = x.min()
            max = x.max()
        */
        let x: Tensor = [1.0, 2.0, 3.0].try_into()?;
        let min = x.min()?;
        let max = x.max()?;
        assert_eq!(min.f64s_ref()?, &[1.0]); // #PYTHON min.tolist()
        assert_eq!(max.f64s_ref()?, &[3.0]); // #PYTHON max.tolist()

        Ok(())
    }
    #[test]
    fn test_flash_powder_flip() -> StableTorchResult<()> {
        /*
            #|PYTHON
            x = torch.tensor(list(range(1,9)), dtype=torch.int64).reshape([2,2,2])
            f = torch.flip(x, [0, 1])
        */
        let d = Tensor::from([1i64, 2, 3, 4, 5, 6, 7, 8])?;
        let x = d.view(&[2, 2, 2])?;
        assert_eq!(x.i64s_ref()?, &[1, 2, 3, 4, 5, 6, 7, 8]); // #PYTHON x.ravel().tolist()
        assert_eq!(x.sizes(), &[2, 2, 2]); // #PYTHON list(x.shape)

        let f = x.flip(&[0, 1])?;
        assert_eq!(f.i64s_ref()?, &[7, 8, 5, 6, 3, 4, 1, 2]); // #PYTHON f.ravel().tolist()
        assert_eq!(f.sizes(), &[2, 2, 2]); // #PYTHON list(f.shape)

        Ok(())
    }
    #[test]
    fn test_flash_powder_ge_tensor() -> StableTorchResult<()> {
        /*
            #|PYTHON
            x = torch.tensor(list(range(1,10)), dtype=torch.int64).reshape([3,3])
            v = torch.tensor(4)
            c = x >= v
        */
        let d = Tensor::from([1i64, 2, 3, 4, 5, 6, 7, 8, 9])?;
        let x = d.view(&[3, 3])?;
        assert_eq!(x.i64s_ref()?, &[1, 2, 3, 4, 5, 6, 7, 8, 9]); // #PYTHON x.ravel().tolist()
        assert_eq!(x.sizes(), &[3, 3]); // #PYTHON list(x.shape)

        let f: Tensor = 4i64.try_into()?;
        let c = x.ge(&f)?;

        assert_eq!(
            c.bools_ref()?,
            &[false, false, false, true, true, true, true, true, true]
        ); // #PYTHON c.ravel().tolist()
        assert_eq!(c.sizes(), &[3, 3]); // #PYTHON list(c.shape)

        Ok(())
    }

    #[test]
    fn test_flash_powder_comparisons_ne_eq_lt_gt_ge_le_tensor_out() -> StableTorchResult<()> {
        /*
            #|PYTHON
            a = torch.tensor([[1, 2], [3, 4]])
            b = torch.tensor([[1, 1], [4, 4]])
            ne = a.ne(b)
            eq = a.eq(b)
            lt = a.lt(b)
            gt = a.gt(b)
            ge = a.ge(b)
            le = a.le(b)
        */
        let a: Tensor = [[1, 2], [3, 4]].try_into()?;
        let b: Tensor = [[1, 1], [4, 4]].try_into()?;
        let ne = a.ne(&b)?;
        assert_eq!(ne.bools_ref()?, &[false, true, true, false]); // #PYTHON ne.ravel().tolist()
        assert_eq!(ne.sizes(), &[2, 2]); // #PYTHON list(ne.shape)

        let eq = a.eq(&b)?;
        assert_eq!(eq.bools_ref()?, &[true, false, false, true]); // #PYTHON eq.ravel().tolist()
        assert_eq!(eq.sizes(), &[2, 2]); // #PYTHON list(eq.shape)

        let lt = a.lt(&b)?;
        assert_eq!(lt.bools_ref()?, &[false, false, true, false]); // #PYTHON lt.ravel().tolist()
        assert_eq!(lt.sizes(), &[2, 2]); // #PYTHON list(lt.shape)

        let gt = a.gt(&b)?;
        assert_eq!(gt.bools_ref()?, &[false, true, false, false]); // #PYTHON gt.ravel().tolist()
        assert_eq!(gt.sizes(), &[2, 2]); // #PYTHON list(gt.shape)

        let ge = a.ge(&b)?;
        assert_eq!(ge.bools_ref()?, &[true, true, false, true]); // #PYTHON ge.ravel().tolist()
        assert_eq!(ge.sizes(), &[2, 2]); // #PYTHON list(ge.shape)

        let le = a.le(&b)?;
        assert_eq!(le.bools_ref()?, &[true, false, true, true]); // #PYTHON le.ravel().tolist()
        assert_eq!(le.sizes(), &[2, 2]); // #PYTHON list(le.shape)

        Ok(())
    }

    #[test]
    fn test_flash_powder_topk() -> StableTorchResult<()> {
        /*
            #|PYTHON
            x = torch.tensor(list(range(1,6)), dtype=torch.int64)
            v = x.topk(3)
        */
        let d = Tensor::from([1i64, 2, 3, 4, 5])?;
        let (values, indices) = d.topk(3, &Default::default())?;
        assert_eq!(values.i64s_ref()?, &[5, 4, 3]); // #PYTHON v.values.ravel().tolist()
        assert_eq!(values.sizes(), &[3]); // #PYTHON list(v.values.shape)
        assert_eq!(indices.i64s_ref()?, &[4, 3, 2]); // #PYTHON v.indices.ravel().tolist()
        assert_eq!(indices.sizes(), &[3]); // #PYTHON list(v.indices.shape)

        /*
            #|PYTHON
            x = torch.tensor(list(range(1,6)), dtype=torch.int64)
            v = x.topk(3, largest=False)
        */
        let (values, indices) = d.topk(
            3,
            &TopKOptions {
                largest: false,
                ..Default::default()
            },
        )?;
        assert_eq!(values.i64s_ref()?, &[1, 2, 3]); // #PYTHON v.values.ravel().tolist()
        assert_eq!(values.sizes(), &[3]); // #PYTHON list(v.values.shape)
        assert_eq!(indices.i64s_ref()?, &[0, 1, 2]); // #PYTHON v.indices.ravel().tolist()
        assert_eq!(indices.sizes(), &[3]); // #PYTHON list(v.indices.shape)

        Ok(())
    }

    #[test]
    fn test_flash_powder_remainder() -> StableTorchResult<()> {
        /*
            #|PYTHON
            a = torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
            b = torch.remainder(torch.tensor([1, 2, 3, 4, 5]), -1.5)
        */
        let a_in: Tensor = [-3.0f32, -2.0, -1.0, 1.0, 2.0, 3.0].try_into()?;
        let a_second: Tensor = 2i64.try_into()?;

        let a = a_in.remainder(&a_second)?;

        assert_eq!(a.f32s_ref()?, &[1.0, -0.0, 1.0, 1.0, 0.0, 1.0]); // #PYTHON a.ravel().tolist()
        assert_eq!(a.sizes(), &[6]); // #PYTHON list(a.shape)

        let b_in: Tensor = [1i64, 2, 3, 4, 5].try_into()?;
        let b_second: Tensor = (-1.5f32).try_into()?;

        let b = b_in.remainder(&b_second)?;

        assert_eq!(b.f32s_ref()?, &[-0.5, -1.0, 0.0, -0.5, -1.0]); // #PYTHON b.ravel().tolist()
        assert_eq!(b.sizes(), &[5]); // #PYTHON list(b.shape)
        Ok(())
    }

    #[test]
    fn test_flash_powder_to_cpu() -> StableTorchResult<()> {
        let a_in: Tensor = [-3.0f32, -2.0, -1.0, 1.0, 2.0, 3.0].try_into()?;

        let a_in_cpu = a_in.cpu()?;
        assert!(a_in.is_equal(&a_in_cpu)?);

        #[cfg(feature = "cuda")]
        {
            let a_in: Tensor = [-3.0f32, -2.0, -1.0, 1.0, 2.0, 3.0].try_into()?;
            let a_cuda = a_in.to(&crate::Device::CUDA.into())?;
            assert_eq!(
                a_cuda.device().device_type(),
                crate::Device::CUDA.device_type()
            );

            let a_in_cpu = a_cuda.cpu()?;
            assert!(a_in.is_equal(&a_in_cpu)?);
        }

        Ok(())
    }
}
