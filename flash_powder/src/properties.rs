//! Retrieve properties like [`sizes`][`TensorProperties::sizes()`] from any [`TensorAccess`].

use crate::{Ten, TenMut, Tensor, TensorAccess, TensorAccessMut, dtype::DType};
use torch_stable::headeronly::core::Layout;
use torch_stable::stable::device::{Device, DeviceIndex};

/// Fundamental tensor properties like size, dimensionality, type etc.
pub trait TensorProperties: TensorAccess {
    fn dim(&self) -> usize {
        self.get_tensor().dim()
    }

    fn numel(&self) -> usize {
        self.get_tensor().numel()
    }

    fn sizes(&self) -> &[usize] {
        self.get_tensor().sizes()
    }

    fn strides(&self) -> &[usize] {
        self.get_tensor().strides()
    }

    fn stride(&self, dim: usize) -> usize {
        self.get_tensor().stride(dim)
    }

    fn is_contiguous(&self) -> bool {
        self.get_tensor().is_contiguous()
    }

    fn dtype(&self) -> DType {
        self.get_tensor().scalar_type().try_into().unwrap()
    }
    fn layout(&self) -> Layout {
        self.get_tensor().layout()
    }

    fn device(&self) -> Device {
        self.get_tensor().device()
    }

    fn device_index(&self) -> DeviceIndex {
        self.get_tensor().get_device_index()
    }

    fn is_cpu(&self) -> bool {
        self.get_tensor().is_cpu()
    }

    fn is_cuda(&self) -> bool {
        self.get_tensor().is_cpu()
    }

    fn size(&self, dim: usize) -> usize {
        self.get_tensor().size(dim)
    }

    /// Exact same as size, but allows for negative indices.
    ///
    /// This follows the python semantics better, but `usize` was chosen to not have to cast lengths all the time.
    fn isize(&self, dim: isize) -> usize {
        self.get_tensor().size(dim as usize)
    }

    fn is_defined(&self) -> bool {
        self.get_tensor().defined()
    }

    fn element_size(&self) -> usize {
        self.get_tensor().element_size()
    }

    /// Returns the storage offset of the tensor.
    ///
    /// In number of elements.
    ///
    /// - [stable docs](https://github.com/pytorch/pytorch/blob/7ee00f187cb55019d648efc6779c0925f643f01c/torch/csrc/stable/tensor_struct.h#L382-L397)
    fn storage_offset(&self) -> usize {
        self.get_tensor().storage_offset()
    }

    // fn data_ptr(&self) -> *const u8 {
    //     self.get_tensor().data_ptr()
    // }

    fn const_data_ptr(&self) -> *const u8 {
        self.get_tensor().const_data_ptr()
    }
}

impl TensorProperties for Tensor {}
impl<'a> TensorProperties for Ten<'a> {}
impl<'a> TensorProperties for TenMut<'a> {}

/// Fundamental mutable tensor properties.
///
/// This is a bit of an odd thing, but this is necessary because [`TensorPropertiesMut::mutable_data_ptr`] needs to have
/// a mutable borrow, but this also materializes a lazily cloned tensor and allows its contents to be changed, so it
/// can't be implemented for [`Ten<'_>`].
///
pub trait TensorPropertiesMut: TensorAccessMut + TensorProperties {
    fn mutable_data_ptr(&mut self) -> *mut u8 {
        self.get_tensor_mut().mutable_data_ptr()
    }
}
impl TensorPropertiesMut for Tensor {}
impl<'a> TensorPropertiesMut for TenMut<'a> {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_flash_powder_indexing() -> crate::StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,13)), dtype=torch.float).reshape([1, 3, 4])
        */

        let d = Tensor::from([[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]])?;
        assert_eq!(d.sizes(), &[1, 3, 4]); // #PYTHON list(d.shape)
        assert_eq!(d.isize(0), 1); // #PYTHON d.size(0)
        assert_eq!(d.isize(1), 3); // #PYTHON d.size(1)
        assert_eq!(d.isize(2), 4); // #PYTHON d.size(2)
        assert_eq!(d.isize(-1), 4); // #PYTHON d.size(-1)
        assert_eq!(d.isize(-2), 3); // #PYTHON d.size(-2)
        assert_eq!(d.isize(-3), 1); // #PYTHON d.size(-3)

        Ok(())
    }
}
