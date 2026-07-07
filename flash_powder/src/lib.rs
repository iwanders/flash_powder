//!
//! Main building blocks:
//!
//! - [`Tensor`]; Owning tensor, this owns the data, created with [`TensorFactory`][factory::TensorFactory]. (think `Vec<u8>`)
//! - [`Ten<'_>`]; Const borrow of Tensor, this has a parent, its lifetime cannot exceed the parent. (think `&[u8]`)
//! - [`TenMut<'_>`]; Mutable borrow of Tensor, this has a mutable parent, its lifetime cannot exceed the parent. (think `&mut [u8]`)
//!
//! All of these provide [`TensorAccess`] and all functions and methods are implemented on that trait.
//!
//! - [`properties::TensorProperties`]: Methods to retrieve tensor properties like dimension and size.
//! - [`data`]`::{`[`DataRef`][`data::DataRef`], [`DataMut`][`data::DataMut`]`}`: Traits to access the tensor's data as bytes or other types.
//! - [`core_methods`]`::`[`CoreMethods`][`core_methods::CoreMethods`]: Methods / Functions on [`TensorAccess`] that require const access.
//! - [`core_methods`]`::`[`CoreMethodsMut`][`core_methods::CoreMethodsMut`]: Methods / Functions on [`TensorAccess`] that require mutable access.
//! - [`functional`]: Holds free functions line [`conv2d`][`functional::conv2d`] and [`relu`][`functional::relu`], just like PyTorch's Functional.
//!
//! Other principles;
//! - No unsafe in the public interface, safe behaviour as you'd expect.
//! - No interior mutability, all methods are const correct.
//! - Modifying one tensor will not modify another, unless through an mutable borrow.
//! - Rust style lifetimes on tensors, either tied together with an explicit lifetime, or completely separate.
//!
//! The [`nn`] module provides pure-Rust implementations for some of Pytorch's nn submodule;
//! - [`nn::Module`]: Main trait for a neural network layer.
//! - [`nn::layer`]: Implemented layers, these are nothing more than structs owning weights and calling into the appropriate [`functional`].
//!

/*

Todo;
    - Figure out how do we want to do overloads??? SHould it take input arguments?
        Example is squeeze_dim; https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L5856
    - Add add, sub, lol.
    - Stacking of tensors?
    - Use into for self transfer
    - Swap V and B
Nice to have:
    - Printing with scientific mode / int mode, see comment in printing.rs
    - Summarized printing without copying the entire tensor to contiguous and cpu, only copy what is printed.

Tricky:
    - Indexing with tensors with "index.Tensor" always returns a copy, but the current indexing system returns a view.
      We can't reconcile this without an extra method, or indexing overload or something. For now we can use index_tensor,
    - Currently reading a statedict doesn't clear optional tensors.

 */

pub mod torch;

pub mod core_methods;
pub mod data;
pub mod factory;
pub mod functional;
pub mod index;
pub mod properties;
pub mod size;
pub mod tensor;

pub mod conversion;
pub mod dtype;
pub mod nn;
pub mod printing;

/// Shorthands to types.
pub use dtype::DType;
pub use tensor::{Ten, TenMut, Tensor, TensorAccess};
pub use torch_stable::StableTorchResult;

pub use torch_stable::headeronly::core::{Layout, MemoryFormat};
pub use torch_stable::stable::device::Device;

pub use torch_stable;

mod f16;

/// The prelude that contains all the necessary traits.
pub mod prelude {
    use super::*;
    #[doc(inline)]
    pub use core_methods::{CoreMethods, CoreMethodsMut};
    #[doc(inline)]
    pub use data::{DataMut, DataRef};
    #[doc(inline)]
    pub use factory::TensorFactory;
    #[doc(inline)]
    pub use properties::TensorProperties;
    // #[doc(inline)]
    // pub use tensor::{Ten, TenMut, Tensor, TensorAccess};

    #[doc(inline)]
    pub use index::{TensorIndex, TensorIndexMut};

    // #[doc(inline)]
    // pub use super::torch;
    // #[doc(inline)]
    // pub use crate::functional;
    //

    // #[doc(inline)]
    // pub use super::dtype::DType;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::factory::TensorFactory;
    use tensor::Tensor;
    pub use torch_stable::StableTorchResult;

    #[test]
    fn test_flash_powder_create_error() -> StableTorchResult<()> {
        let a = Tensor::zeros(&[usize::MAX, 5], &Default::default());

        assert!(a.is_err());
        let failure = a.err().unwrap();
        let v = failure.to_string();
        println!("v: {v}");
        assert!(v.contains("(zeros: Dimension size must be non-negative.)"));

        Ok(())
    }
}
