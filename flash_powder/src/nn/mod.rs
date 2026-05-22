//! Torch's nn module.
//!
//! <https://docs.pytorch.org/docs/2.12/nn.html>
//!
//! This is not exposed at all through the stable API, so this is a pure rust implementation.

mod module;
pub use module::*;
mod layers;
pub use layers::*;
