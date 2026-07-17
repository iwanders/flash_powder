//! Torch's nn module.
//!
//! <https://docs.pytorch.org/docs/2.12/nn.html>
//!
//! This is not exposed at all through the stable API, so this is a pure rust implementation.
//!
//! Core trait is [`Module`], which also provides plumbing to tensors through [`StateDictReader`].

pub mod functional;
pub mod layer;
pub mod module;
pub use layer::*;
pub use module::*;
