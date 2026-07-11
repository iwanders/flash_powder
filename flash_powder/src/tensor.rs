//! Holds the three Tensor types.
use crate::StableTorchResult;
use crate::core_methods::CoreMethods;
use anyhow;
use torch_stable::stable::tensor::Tensor as StableTensor;

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
