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
use crate::properties::TensorProperties;
use crate::size::Size;
use crate::{StableTorchResult, Ten, TenMut, Tensor, TensorAccess};
use torch_stable::{
    aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor, unsafe_call_bail,
    unsafe_call_dispatch_bail,
};
use torch_stable::{aoti_torch::*, unsafe_call_dispatch_panic};

use torch_stable::headeronly::core::MemoryFormat;
#[derive(Copy, Clone, Debug)]
#[derive(Default)]
pub struct MeanOptions {
    pub dim: Option<usize>,
    pub keepdim: bool,
    pub dtype: Option<DType>,
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
        Ok(Ten::new(self.get_tensor(), stack[0].try_into()?))
    }

    /// To
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L8033)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.to.html)
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
        assert_ne!(self.data_ptr(), r.data_ptr());

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
        assert_eq!(self.data_ptr(), r.data_ptr());
        Ok(Ten::new(self.get_tensor(), r))
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
    fn equal<T: TensorAccess>(&self, other: &T) -> StableTorchResult<bool> {
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

    /// Lazily clone this into an owning tensor.
    fn to_owned(&self) -> StableTorchResult<Tensor> {
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

    /// Permute
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4675)
    /// - [tensor method](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.permute.html)
    /// - [pytorch method](https://docs.pytorch.org/docs/2.11/generated/torch.permute.html#torch-permute)
    fn permute(&self, dims: &[usize]) -> StableTorchResult<Ten<'_>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), dims.into()];
        unsafe_call_dispatch_bail!("aten::permute", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.data_ptr(), r.data_ptr());
        Ok(Ten::new(self.get_tensor(), r))
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
}

/// Core methods that require mutable access.
///
/// See the [`core_methods`][crate::core_methods] module for description of this trait's functionality.
pub trait CoreMethodsMut: TensorAccess + TensorProperties {
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
        assert_eq!(retrieve.data_ptr(), self.data_ptr());
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
        assert_eq!(self.data_ptr(), r.data_ptr());
        Ok(TenMut::new(self.get_tensor_mut(), r))
    }

    fn ten_mut<'a>(&'a mut self) -> StableTorchResult<TenMut<'a>> {
        let shape = self.sizes();
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), (shape).into()];
        unsafe_call_dispatch_bail!("aten::view", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.data_ptr(), r.data_ptr());
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
        assert_eq!(self.data_ptr(), r.data_ptr());
        Ok(TenMut::new(self.get_tensor_mut(), r))
    }
}
impl CoreMethodsMut for Tensor {}
impl<'a> CoreMethodsMut for Ten<'a> {}
impl<'a> CoreMethodsMut for TenMut<'a> {}

impl<'a> TenMut<'a> {
    /// Narrow mut view
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.narrow.html)
    pub fn narrow_mut(
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

    pub fn select_mut(self, dim: usize, index: usize) -> StableTorchResult<TenMut<'a>> {
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

        let d = Tensor::from(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])?;
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
            d = torch.tensor(d)
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

        let mut n = a.to_owned()?;
        // Currently lazy copy
        let old_n_ptr = n.const_data_ptr();
        assert_eq!(n.const_data_ptr(), a.const_data_ptr());
        assert!(n.equal(&a)?);

        // Verify n holds same data
        assert_eq!(n.f32s_ref()?[0], 50.0);
        // Modify n, this performs the copy.
        n.f32s_mut()?[0] = 20.0;
        assert_eq!(n.equal(&a)?, false);

        // data pointer shouldn't be the same now.
        assert_ne!(n.const_data_ptr(), old_n_ptr);

        assert_eq!(d.f32s_mut()?[0], 50.0);

        // Try a non owning view
        let v = d.view(&[16])?;
        let mut cv = v.to_owned()?;
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

        let d = Tensor::from(&[[
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
        let d = Tensor::from(&[[
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

        let d = Tensor::from(&[[
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
        assert_eq!(v.equal(&v_c)?, true);

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

        let t = Tensor::from(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])?;
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

        let t = Tensor::from(&[0.3810f32, 1.2774, -0.2972, -0.3719, 0.4637])?;
        let denom: Tensor = (0.5f32,).try_into()?;
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
    fn test_flash_powder_mul() -> StableTorchResult<()> {
        // https://docs.pytorch.org/docs/2.11/generated/torch.mul.html#torch.mul
        /*
            #|PYTHON
            x = torch.tensor([ 0.2015, -0.4255,  2.6087])
            r = torch.mul(x, 100.0)
        */

        let t = Tensor::from(&[0.2015f32, -0.4255, 2.6087])?;
        let factor: Tensor = (100.0f32,).try_into()?;
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
}
