//! Accessors to the data in the Tensor.
//!
//! - These only work if the Tensor is on the cpu.
//! - These methods only work if the tensor is contiguous ([`is_contiguous()`][`TensorProperties::is_contiguous`] is `true`).
//!
//! Come in three flavours:
//! - `<T>s_ref()`, slice to the entire tensor, example: [`f32s_ref()`][`DataRef::f32s_ref`]
//! - `<T>_ref(indices: &[usize])`, indexed access to a single scalar, requires [`dim()`][`TensorProperties::dim`]` == indices.len()`, example: [`f32_ref()`][`DataRef::f32_ref`]
//! - `as_<T>()`, indexing for as a scalar, same as `<T>_ref(&[])`, example: [`as_f32()`][`DataRef::as_f32`]
//!
//! The access type is checked against the tensor's [`dtype()`][`TensorProperties::dtype`], if you don't want that use [`data()`][`DataRef::data`] and [`data_mut()`][`DataMut::data_mut`].
use anyhow::bail;
use zerocopy::{Immutable, IntoBytes, KnownLayout, TryFromBytes};

use crate::dtype::ScalarDType;
use crate::{
    properties::TensorProperties,
    tensor::{Ten, TenMut, Tensor, TensorAccess, TensorAccessMut},
};
use torch_stable::{StableTorchResult, contrib::TensorPropertiesContrib as _};

macro_rules! impl_slice_ref {
    ($t:ty, $v:ident) => {
        /// Access the entire tensor's slice using this datatype.
        fn $v(&self) -> StableTorchResult<&[$t]> {
            self.ds_ref::<$t>()
        }
    };
}
macro_rules! impl_item_ref {
    ($t:ty, $v:ident) => {
        /// Indexed access to a single value of type, requires [`dim()`][`TensorProperties::dim`]` == indices.len()`
        fn $v(&self, indices: &[usize]) -> StableTorchResult<&$t> {
            self.d_ref::<$t>(indices)
        }
    };
}
macro_rules! impl_as_ref {
    ($t:ty, $v:ident) => {
        /// Access the tensor as a scalar, requires [`numel()`][`TensorProperties::numel`]` == 1`
        fn $v(&self) -> StableTorchResult<&$t> {
            if self.numel() != 1 {
                bail!(
                    "can only use as_<T> with 0 dimensional tensors, dim was {}",
                    self.dim()
                )
            }
            self.d_ref::<$t>(&[])
        }
    };
}

/// Access data in the tensor through a reference.
///
/// Methods ending with 's' return a slice.
///
/// These methods are only valid if the data is on the CPU.
pub trait DataRef: TensorAccess + TensorProperties {
    /// Direct access to the byte slice backing the tensor.
    fn data(&self) -> StableTorchResult<&[u8]> {
        if !self.is_contiguous() {
            bail!("cannot get slice into non contiguous tensor");
        }

        let z = self.get_tensor();
        // https://github.com/pytorch/pytorch/blob/ec673ecd/c10/core/TensorImpl.h#L743-L756
        let data_ptr = z.const_data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts(data_ptr, z.storage_size()) })
        } else {
            bail!(
                "tensor is on {:?}, move to the cpu to access slice",
                z.device()
            )
        }
    }

    /// Access to the slice spanning the entire tensor data.
    ///
    /// Zerocopy cast of [`Self::u8s_ref`] to `&[T]`.
    fn ds_ref<T: IntoBytes + TryFromBytes + Immutable + ScalarDType>(
        &self,
    ) -> StableTorchResult<&[T]> {
        if T::type_dtype() != self.dtype() {
            bail!(
                "accessing tensor with dtype {:?} using {:?}",
                self.dtype(),
                T::type_dtype()
            );
        }
        let byte_ref = self.data()?;
        match <[T]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    /// Element reference to element at the provided indices.
    fn d_ref<T: IntoBytes + TryFromBytes + Immutable + KnownLayout + ScalarDType>(
        &self,
        index: &[usize],
    ) -> StableTorchResult<&T> {
        if T::type_dtype() != self.dtype() {
            bail!(
                "accessing tensor with dtype {:?} using {:?}",
                self.dtype(),
                T::type_dtype()
            );
        }
        if index.len() > self.dim() {
            bail!(
                "indices provided {} dim, tensor is {} dim",
                index.len(),
                self.dim()
            )
        }

        let mut offset = 0;
        for (dim, index) in index.iter().enumerate() {
            if *index >= self.sizes()[dim] {
                bail!(
                    "index {} for dimension {} exceeded size {}",
                    index,
                    dim,
                    self.sizes()[dim]
                );
            }
            offset += self.stride(dim) * index;
        }
        let size = std::mem::size_of::<T>();
        if std::mem::size_of::<T>() != self.element_size() {
            bail!(
                "indexing with element size of {} which does not match {}",
                std::mem::size_of::<T>(),
                self.element_size()
            )
        }
        let byte_ref = &self.data()?[offset * size..size * (offset + 1)];
        match <T>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    // Slices
    impl_slice_ref!(f32, f32s_ref);
    impl_slice_ref!(f64, f64s_ref);

    impl_slice_ref!(u8, u8s_ref);
    impl_slice_ref!(u16, u16s_ref);
    impl_slice_ref!(u32, u32s_ref);
    impl_slice_ref!(u64, u64s_ref);

    impl_slice_ref!(i8, i8s_ref);
    impl_slice_ref!(i16, i16s_ref);
    impl_slice_ref!(i32, i32s_ref);
    impl_slice_ref!(i64, i64s_ref);

    impl_slice_ref!(bool, bools_ref);

    // Indexed
    impl_item_ref!(f32, f32_ref);
    impl_item_ref!(f64, f64_ref);

    impl_item_ref!(u8, u8_ref);
    impl_item_ref!(u16, u16_ref);
    impl_item_ref!(u32, u32_ref);
    impl_item_ref!(u64, u64_ref);

    impl_item_ref!(i8, i8_ref);
    impl_item_ref!(i16, i16_ref);
    impl_item_ref!(i32, i32_ref);
    impl_item_ref!(i64, i64_ref);

    impl_slice_ref!(bool, bool_ref);

    // Scalar
    impl_as_ref!(f32, as_f32);
    impl_as_ref!(f64, as_f64);

    impl_as_ref!(u8, as_u8);
    impl_as_ref!(u16, as_u16);
    impl_as_ref!(u32, as_u32);
    impl_as_ref!(u64, as_u64);

    impl_as_ref!(i8, as_i8);
    impl_as_ref!(i16, as_i16);
    impl_as_ref!(i32, as_i32);
    impl_as_ref!(i64, as_i64);

    impl_slice_ref!(bool, as_bool);
}

impl DataRef for Tensor {}
impl<'a> DataRef for Ten<'a> {}
impl<'a> DataRef for TenMut<'a> {}

macro_rules! impl_slice_mut {
    ($t:ty, $v:ident) => {
        /// Access the entire tensor's mutable slice using this datatype.
        fn $v(&mut self) -> StableTorchResult<&mut [$t]> {
            self.ds_mut::<$t>()
        }
    };
}
macro_rules! impl_item_mut {
    ($t:ty, $v:ident) => {
        /// Indexed access to a single mutable value of type, requires [`dim()`][`TensorProperties::dim`]` == indices.len()`
        fn $v(&mut self, indices: &[usize]) -> StableTorchResult<&mut $t> {
            self.d_mut::<$t>(indices)
        }
    };
}

macro_rules! impl_as_mut {
    ($t:ty, $v:ident) => {
        /// Access the tensor as a mutable scalar, requires [`dim()`][`TensorProperties::dim`]` == 0`
        fn $v(&mut self) -> StableTorchResult<&mut $t> {
            if self.dim() != 0 {
                bail!(
                    "can only use as_<T> with 0 dimensional tensors, dim was {}",
                    self.dim()
                )
            }
            self.d_mut::<$t>(&[])
        }
    };
}

/// Access data in the tensor through a mutable reference.
///
/// Methods ending with 's' return a slice.
///
/// These methods are only valid if the data is on the CPU.
pub trait DataMut: TensorAccessMut + TensorProperties {
    /// Direct access to the mutable byte slice backing the tensor.
    fn data_mut(&mut self) -> StableTorchResult<&mut [u8]> {
        if !self.is_contiguous() {
            bail!("cannot get slice into non contiguous tensor");
        }
        let z = self.get_tensor_mut();
        // https://github.com/pytorch/pytorch/blob/ec673ecd/c10/core/TensorImpl.h#L743-L756
        let data_ptr = z.mutable_data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts_mut(data_ptr, z.storage_size()) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    /// Access to the mutable slice spanning the entire tensor data.
    ///
    /// Cast of [`Self::data_mut`] to `&mut [T]`.
    fn ds_mut<T: IntoBytes + TryFromBytes + Immutable + ScalarDType>(
        &mut self,
    ) -> StableTorchResult<&mut [T]> {
        if T::type_dtype() != self.dtype() {
            bail!(
                "accessing tensor with dtype {:?} using {:?}",
                self.dtype(),
                T::type_dtype()
            );
        }
        let byte_ref = self.data_mut()?;
        match <[T]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    /// Element mutable reference to element at the provided indices.
    fn d_mut<T: IntoBytes + TryFromBytes + Immutable + KnownLayout + ScalarDType>(
        &mut self,
        index: &[usize],
    ) -> StableTorchResult<&mut T> {
        if T::type_dtype() != self.dtype() {
            bail!(
                "accessing tensor with dtype {:?} using {:?}",
                self.dtype(),
                T::type_dtype()
            );
        }
        if index.len() > self.dim() {
            bail!(
                "indices provided {} dim, tensor is {} dim",
                index.len(),
                self.dim()
            )
        }

        let mut offset = 0;
        for (dim, index) in index.iter().enumerate() {
            if *index >= self.sizes()[dim] {
                bail!(
                    "index {} for dimension {} exceeded size {}",
                    index,
                    dim,
                    self.sizes()[dim]
                );
            }
            offset += self.stride(dim) * index;
        }
        if std::mem::size_of::<T>() != self.element_size() {
            bail!(
                "indexing with element size of {} which does not match {}",
                std::mem::size_of::<T>(),
                self.element_size()
            )
        }
        let size = std::mem::size_of::<T>();
        let byte_mut = &mut self.data_mut()?[offset * size..size * (offset + 1)];
        match <T>::try_mut_from_bytes(byte_mut) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    // Slices
    impl_slice_mut!(f32, f32s_mut);
    impl_slice_mut!(f64, f64s_mut);

    impl_slice_mut!(u16, u16s_mut);
    impl_slice_mut!(u32, u32s_mut);
    impl_slice_mut!(u64, u64s_mut);

    impl_slice_mut!(i8, i8s_mut);
    impl_slice_mut!(i16, i16s_mut);
    impl_slice_mut!(i32, i32s_mut);
    impl_slice_mut!(i64, i64s_mut);

    // Indexed
    impl_item_mut!(f32, f32_mut);
    impl_item_mut!(f64, f64_mut);

    impl_item_mut!(u8, u8_mut);
    impl_item_mut!(u16, u16_mut);
    impl_item_mut!(u32, u32_mut);
    impl_item_mut!(u64, u64_mut);

    impl_item_mut!(i8, i8_mut);
    impl_item_mut!(i16, i16_mut);
    impl_item_mut!(i32, i32_mut);
    impl_item_mut!(i64, i64_mut);

    // Scalar
    impl_as_mut!(f32, as_f32_mut);
    impl_as_mut!(f64, as_f64_mut);

    impl_as_mut!(u8, as_u8_mut);
    impl_as_mut!(u16, as_u16_mut);
    impl_as_mut!(u32, as_u32_mut);
    impl_as_mut!(u64, as_u64_mut);

    impl_as_mut!(i8, as_i8_mut);
    impl_as_mut!(i16, as_i16_mut);
    impl_as_mut!(i32, as_i32_mut);
    impl_as_mut!(i64, as_i64_mut);
}
impl DataMut for Tensor {}
impl<'a> DataMut for TenMut<'a> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{index::TensorIndexMut as _, prelude::*};

    #[test]
    fn test_flash_powder_data_index() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([ 4,4])
            z = torch.narrow(d, 1, 0, 3)
        */

        let mut d = Tensor::from([
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ])?;
        assert_eq!(d.sizes(), &[4, 4]); // #PYTHON list(d.shape)
        assert_eq!(
            d.f32s_ref()?,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())

        assert_eq!(d.f32_ref(&[2, 2])?, &11.0); // #PYTHON d[2, 2].item()
        assert_eq!(d.f32_ref(&[2, 3])?, &12.0); // #PYTHON d[2, 3].item()
        assert_eq!(d.f32_ref(&[0, 3])?, &4.0); // #PYTHON d[0, 3].item()
        assert_eq!(d.f32_ref(&[3, 0])?, &13.0); // #PYTHON d[3, 0].item()

        println!("d: {d:?}");
        let mut z = d.narrow_mut(1, 0, 3)?;
        println!("z: {z:?}");

        assert_eq!(z.sizes(), &[4, 3]); // #PYTHON list(z.shape)
        assert_eq!(z.is_contiguous(), false);
        assert_eq!(z.storage_offset(), 0);

        assert_eq!(z.i((0, 0))?.as_f32()?, &1.0); // #PYTHON z[ 0,  0].item()
        assert_eq!(z.i((0, 1))?.as_f32()?, &2.0); // #PYTHON z[ 0,  1].item()
        assert_eq!(z.i((0, 2))?.as_f32()?, &3.0); // #PYTHON z[ 0,  2].item()
        assert_eq!(z.i((1, 0))?.as_f32()?, &5.0); // #PYTHON z[ 1,  0].item()
        assert_eq!(z.i((1, 1))?.as_f32()?, &6.0); // #PYTHON z[ 1,  1].item()
        assert_eq!(z.i((1, 2))?.as_f32()?, &7.0); // #PYTHON z[ 1,  2].item()

        // What about as scalar on a non 0 dim tensor:
        let scalar_1d: Tensor = [3.3f32].try_into()?;
        assert_eq!(scalar_1d.as_f32()?, &3.3);

        /*
            #|PYTHON
            z[0,2] = 50.0
            z[1,2] = 100.0
        */

        *(z.i_mut((0, 2))?.as_f32_mut()?) = 50.0;
        *(z.i_mut((1, 2))?.as_f32_mut()?) = 100.0;

        assert_eq!(z.i((0, 2))?.as_f32()?, &50.0); // #PYTHON z[ 0,  2].item()
        assert_eq!(z.i((1, 2))?.as_f32()?, &100.0); // #PYTHON z[ 1,  2].item()

        /*
            #|PYTHON
            y = torch.narrow(d, 1, 1, 3)
        */

        // Do something with a storage offset.
        let y = d.narrow(1, 1, 3)?;
        println!("y: {y:?}");
        assert_eq!(y.sizes(), &[4, 3]); // #PYTHON list(y.shape)
        assert_eq!(y.is_contiguous(), false);
        assert_eq!(y.storage_offset(), 1);
        assert_eq!(y.i((0, 2))?.as_f32()?, &4.0); // #PYTHON y[ 0,  2].item()
        assert_eq!(y.i((1, 2))?.as_f32()?, &8.0); // #PYTHON y[ 1,  2].item()

        /*
            #|PYTHON
            x = torch.narrow(d, 1, 1, 3).narrow(0, 1, 3)
        */

        // Do something with a storage offset.
        let x = d.narrow(1, 1, 3)?.narrow(0, 1, 3)?;
        println!("x: {x:?}");
        assert_eq!(x.sizes(), &[3, 3]); // #PYTHON list(x.shape)
        assert_eq!(x.is_contiguous(), false);
        assert_eq!(x.storage_offset(), 5);
        assert_eq!(x.i((0, 2))?.as_f32()?, &8.0); // #PYTHON x[ 0,  2].item()
        assert_eq!(x.i((1, 2))?.as_f32()?, &12.0); // #PYTHON x[ 1,  2].item()
        Ok(())
    }

    #[test]
    fn test_flash_powder_permute_data_index_issue() -> StableTorchResult<()> {
        // Test all flavours in depth.
        let mut x = Tensor::randn(&[2, 3, 5], &Default::default())?;
        let s = x.shape();
        for c in 0..s[0] {
            for h in 0..s[1] {
                for w in 0..s[2] {
                    assert!(x.f32_mut(&[c, h, w]).is_ok())
                }
            }
        }
        let x = Tensor::randn(&[2, 3, 5], &Default::default())?;
        let s = x.shape();
        for c in 0..s[0] {
            for h in 0..s[1] {
                for w in 0..s[2] {
                    assert!(x.f32_ref(&[c, h, w]).is_ok())
                }
            }
        }
        let x = Tensor::randn(&[2, 3, 5], &Default::default())?;
        let y = x.permute(&[2, 0, 1])?;
        let s = y.shape();
        assert_eq!(&s, &[5, 2, 3]);
        assert_eq!(y.is_contiguous(), false);
        let mut x = Tensor::randn(&[2, 3, 5], &Default::default())?;
        let y = x.permute_mut(&[2, 0, 1])?;
        let s = y.shape();
        assert_eq!(&s, &[5, 2, 3]);
        assert_eq!(y.is_contiguous(), false);

        Ok(())
    }
}
