//! Holds the three Tensor types.
use crate::StableTorchResult;
use anyhow;
use torch_stable::unsafe_call_dispatch_panic;
use torch_stable::{aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor};

/// A tensor, this owns its data.
///
/// Interact with it through any of the traits that are implemented for [`TensorAccess`].
///
/// Usually you don't create this directly, but create tensors through [`crate::factory::TensorFactory`].
pub struct Tensor {
    tensor: StableTensor,
}
impl Clone for Tensor {
    /// This is a full owning clone, but lazy.
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
    fn clone(&self) -> Self {
        // Clone cannot throw... so we use a lazy clone; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1278
        let mut stack: [StableIValue; 1] = [(&self.tensor).into()];
        unsafe_call_dispatch_panic!("aten::_lazy_clone", "", stack.as_mut_slice());
        let r: Tensor = Self::new(stack[0].try_into().unwrap());

        r
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
    _parent: &'a StableTensor,
}
impl<'a> Ten<'a> {
    pub fn new(parent: &'a StableTensor, tensor: StableTensor) -> Self {
        Self {
            _parent: parent,
            tensor,
        }
    }
    pub(crate) fn as_parent(&self) -> &'a StableTensor {
        self._parent
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

pub trait TensorAccess {
    fn get_tensor(&self) -> &StableTensor;
    fn get_tensor_mut(&mut self) -> &mut StableTensor;
}

impl<'a> TensorAccess for TenMut<'a> {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        &mut self.tensor
    }
}

impl<'a> TensorAccess for Ten<'a> {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        &mut self.tensor
    }
}

impl TensorAccess for Tensor {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        &mut self.tensor
    }
}

impl TensorAccess for &Tensor {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
    // This here is a mutable reference but an immutable tensor :<
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        unreachable!(
            "this is here because I didn't split out TensorAccess in mut and non mut... this should never be called"
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_flash_powder_torch_select() -> StableTorchResult<()> {
        let a: Tensor = 3.try_into()?;
        let mut b: &Tensor = &a;

        // Bah, this actually compiles :<
        //let m = b.get_tensor_mut();

        Ok(())
    }
}
