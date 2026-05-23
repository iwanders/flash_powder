//! Module with [`TryInto<Tensor>`] implementations.
//!
//! They're not visible in the docs, but [`TryInto<Tensor>`] is implemented for:
//! - `[T]` Creates a 1d tensor.
//! - `[T; N]` Creates a 1d tensor.
//! - `[[T; C]; R]` Creates a 2d tensor.
//! - `[[[T; C]; R]; D]` Creates a 3d tensor.
//!
//! ```rust
//! # use flash_powder::prelude::*;
//! # use flash_powder::{StableTorchResult, Tensor};
//! # fn foo() -> StableTorchResult<()>{
//!
//!   let d: Tensor = (5i64,).try_into()?;
//!   assert_eq!(d.dim(), 0);
//!   assert_eq!(d.i64_ref(&[])?, &5);
//!
//!   let d: Tensor = [5i64, 3].try_into()?;
//!   assert_eq!(d.sizes(), &[2]);
//!   let d: Tensor = [[5.0f32, 3.0], [1.0, 2.0]].try_into()?;
//!   assert_eq!(d.sizes(), &[2, 2]);
//!
//!   let d: Tensor = [[5.0f32, 3.0, 5.0], [1.0, 2.0, 0.0]].try_into()?;
//!   assert_eq!(d.sizes(), &[2, 3]);
//!
//!
//!   let d: Tensor = [[[1i64, 2], [3, 4]], [[8, 1], [9, 3]]].try_into()?;
//!   assert_eq!(d.sizes(), &[2, 2, 2]);
//! # Ok(())
//! # }
//! ```
//!

use crate::factory::TensorFactory;
use crate::tensor::Tensor;
use crate::{data::DataMut, factory::EmptyOptions};

use crate::dtype::ScalarDType;
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

impl<T: ScalarDType + Immutable + IntoBytes + TryFromBytes + Copy> TryInto<Tensor> for (T,) {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[],
            &EmptyOptions {
                dtype: Some(T::type_dtype()),
                ..Default::default()
            },
        )?;
        v.ds_mut::<T>()?[0] = self.0;
        Ok(v)
    }
}

impl<T: ScalarDType + Immutable + IntoBytes + TryFromBytes + Copy> TryInto<Tensor> for &[T] {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[self.len()],
            &EmptyOptions {
                dtype: Some(T::type_dtype()),
                ..Default::default()
            },
        )?;
        v.ds_mut::<T>()?.copy_from_slice(self);
        Ok(v)
    }
}

impl<T: ScalarDType + Immutable + IntoBytes + TryFromBytes + Copy, const V: usize> TryInto<Tensor>
    for [T; V]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[V],
            &EmptyOptions {
                dtype: Some(T::type_dtype()),
                ..Default::default()
            },
        )?;
        v.ds_mut::<T>()?.copy_from_slice(&self);
        Ok(v)
    }
}
// And its ref;
impl<T: ScalarDType + Immutable + IntoBytes + TryFromBytes + Copy, const V: usize> TryInto<Tensor>
    for &[T; V]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[V],
            &EmptyOptions {
                dtype: Some(T::type_dtype()),
                ..Default::default()
            },
        )?;
        v.ds_mut::<T>()?.copy_from_slice(self);
        Ok(v)
    }
}

impl<T: ScalarDType + Immutable + IntoBytes + TryFromBytes + Copy, const C: usize, const R: usize>
    TryInto<Tensor> for [[T; C]; R]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[R, C],
            &EmptyOptions {
                dtype: Some(T::type_dtype()),
                ..Default::default()
            },
        )?;
        v.data_mut()?.copy_from_slice(self.as_bytes());
        Ok(v)
    }
}
// and its ref;
impl<T: ScalarDType + Immutable + IntoBytes + TryFromBytes + Copy, const C: usize, const R: usize>
    TryInto<Tensor> for &[[T; C]; R]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[R, C],
            &EmptyOptions {
                dtype: Some(T::type_dtype()),
                ..Default::default()
            },
        )?;
        v.data_mut()?.copy_from_slice(self.as_bytes());
        Ok(v)
    }
}

impl<
    T: ScalarDType + Immutable + IntoBytes + TryFromBytes + Copy,
    const C: usize,
    const R: usize,
    const D: usize,
> TryInto<Tensor> for [[[T; C]; R]; D]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[D, R, C],
            &EmptyOptions {
                dtype: Some(T::type_dtype()),
                ..Default::default()
            },
        )?;
        v.data_mut()?.copy_from_slice(self.as_bytes());
        Ok(v)
    }
}
// and its ref;
impl<
    T: ScalarDType + Immutable + IntoBytes + TryFromBytes + Copy,
    const C: usize,
    const R: usize,
    const D: usize,
> TryInto<Tensor> for &[[[T; C]; R]; D]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[D, R, C],
            &EmptyOptions {
                dtype: Some(T::type_dtype()),
                ..Default::default()
            },
        )?;
        v.data_mut()?.copy_from_slice(self.as_bytes());
        Ok(v)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::StableTorchResult;
    use crate::data::DataRef;
    use crate::dtype::DType;
    use crate::properties::TensorProperties;

    #[test]
    fn test_tensor_try_from() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor([5, 3])
        */
        let d: Tensor = [5i64, 3].try_into()?;
        assert_eq!(d.sizes(), &[2]); // #PYTHON list(d.shape)
        assert_eq!(d.data()?, &[5, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.dtype(), DType::I64); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([5.0, 3.0])
        */
        let d: Tensor = (&[5.0f32, 3.0]).try_into()?;
        assert_eq!(d.sizes(), &[2]); // #PYTHON list(d.shape)
        assert_eq!(d.data()?, &[0, 0, 160, 64, 0, 0, 64, 64]); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.dtype(), DType::F32); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([[5.0, 3.0], [1.0, 2.0]])
        */
        let d: Tensor = [[5.0f32, 3.0], [1.0, 2.0]].try_into()?;
        assert_eq!(d.sizes(), &[2, 2]); // #PYTHON list(d.shape)
        assert_eq!(
            d.data()?,
            &[0, 0, 160, 64, 0, 0, 64, 64, 0, 0, 128, 63, 0, 0, 0, 64]
        ); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.dtype(), DType::F32); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([1, 3, 4, 5], dtype=torch.int8)
        */
        let d: Tensor = [1i8, 3, 4, 5].try_into()?;
        assert_eq!(d.sizes(), &[4]); // #PYTHON list(d.shape)
        assert_eq!(d.data()?, &[1, 3, 4, 5]); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.dtype(), DType::I8); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([[[5]]])
        */
        let d: Tensor = [[[5i64]]].try_into()?;
        assert_eq!(d.sizes(), &[1, 1, 1]); // #PYTHON list(d.shape)
        assert_eq!(d.data()?, &[5, 0, 0, 0, 0, 0, 0, 0]); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.dtype(), DType::I64); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([True, False, True])
        */
        let d: Tensor = [true, false, true].try_into()?;
        assert_eq!(d.sizes(), &[3]); // #PYTHON list(d.shape)
        assert_eq!(d.data()?, &[1, 0, 1]); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.dtype(), DType::Bool); // #PYTHON d.dtype

        // Non square
        /*
            #|PYTHON
            d = torch.tensor([[5.0, 3.0, 5.0], [1.0, 2.0, 0.0]])
        */
        let d: Tensor = [[5.0f32, 3.0, 5.0], [1.0, 2.0, 0.0]].try_into()?;
        assert_eq!(d.sizes(), &[2, 3]); // #PYTHON list(d.shape)
        assert_eq!(
            d.data()?,
            &[
                0, 0, 160, 64, 0, 0, 64, 64, 0, 0, 160, 64, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 0, 0
            ]
        ); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.dtype(), DType::F32); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([[5.0, 3.0], [1.0, 2.0], [1.0, 2.0]])
        */
        let d: Tensor = [[5.0f32, 3.0], [1.0, 2.0], [1.0, 2.0]].try_into()?;
        assert_eq!(d.sizes(), &[3, 2]); // #PYTHON list(d.shape)
        assert_eq!(
            d.data()?,
            &[
                0, 0, 160, 64, 0, 0, 64, 64, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 128, 63, 0, 0, 0, 64
            ]
        ); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.dtype(), DType::F32); // #PYTHON d.dtype

        // And with depth;
        /*
            #|PYTHON
            d = torch.tensor([[[1, 2],[3,4]], [[8, 1],[9,3]]])
        */
        let d: Tensor = [[[1i64, 2], [3, 4]], [[8, 1], [9, 3]]].try_into()?;
        assert_eq!(d.sizes(), &[2, 2, 2]); // #PYTHON list(d.shape)
        assert_eq!(
            d.data()?,
            &[
                1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
                0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0,
                3, 0, 0, 0, 0, 0, 0, 0
            ]
        ); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.dtype(), DType::I64); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor(int(5))
        */
        let d: Tensor = (5i64,).try_into()?;
        assert_eq!(d.dim(), 0); // #PYTHON d.dim()
        assert_eq!(d.i64_ref(&[])?, &5); // #PYTHON d.item()
        assert_eq!(d.dtype(), DType::I64); // #PYTHON d.dtype

        Ok(())
    }
}
