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
//!
//!
//! Other principles;
//! - No unsafe in the public interface, safe behaviour as you'd expect.
//! - No interior mutability, all methods are const correct.
//! - Modifying one tensor will not modify another, unless through an mutable borrow.
//! - Rust style lifetimes on tensors, either tied together with an explicit lifetime, or completely separate.
//!
//! The [`nn`] module provides pure-Rust implementations for some of Pytorch's nn submodule;
//! - [`nn::Module`]: Main trait for a neural network layer.
//! - [`nn::layer`]: Implemented layers, these are nothing more than structs owning weights and calling into the appropriate [`nn::functional`].
//! - [`nn::functional`]: Holds free functions line [`conv2d`][`nn::functional::conv2d`] and [`relu`][`nn::functional::relu`], just like PyTorch's Functional.
//!

/*

Todo;
    - Figure out how do we want to do overloads??? SHould it take input arguments?
        - Example is squeeze_dim; https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L5856
        - mean(tensor) https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4041
        - and mean(&self, mean_options: &MeanOptions) are already ruined :(
        - Since overloads can always be added, we should probably always just use the full name with _ in between?

Nice to have:
    - Printing with scientific mode / int mode, see comment in printing.rs
    - Summarized printing without copying the entire tensor to contiguous and cpu, only copy what is printed.
    - Note on >= operator;
        // This function has like 5 overloads, the most important are Scalar and Tensor, for now we require TensorAccess
        // in the future, after Scalar is created, we can drop that req in Favour of a ScalarOrTensor trait, which would
        // allow us to handle both with the same function, and also support casting native types to Scalar.
        fn ge<T: TensorAccess + Into<StableIValue>>(&self, other: &T) -> StableTorchResult<Tensor> {


Tricky:
    - Indexing with tensors with "index.Tensor" always returns a copy, but the current indexing system returns a view.
      We can't reconcile this without an extra method, or indexing overload or something. For now we can use index_tensor,

 */

pub mod torch;

pub mod core_methods;
pub mod data;
pub mod factory;
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
pub use tensor::{Ten, TenMut, Tensor, TensorAccess, TensorAccessMut};
pub use torch_stable::StableTorchResult;

pub use torch_stable::headeronly::core::{Layout, MemoryFormat};
pub use torch_stable::stable::device::Device;

pub use torch_stable;

// Yes, this is private, it's only used for printing and not fully featured.
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
    pub use properties::{TensorProperties, TensorPropertiesMut};
    // #[doc(inline)]
    // pub use tensor::{Ten, TenMut, Tensor, TensorAccess};

    #[doc(inline)]
    pub use index::{TensorIndex, TensorIndexMut};

    #[doc(inline)]
    pub use nn::{Module, StateDictAdaptor, StateDictReader};
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
