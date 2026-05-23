//! An owning Size.
//!
//! Why do we need this? Well because .sizes() -> &[usize], which is not owned.
//! This also means that calling sizes() borrows the tensor, this makes it impossible to call a mutable method.
//! In short, the following will not compile:
//! ```ignore
//! # use flash_powder::prelude::*;
//! # use flash_powder::Tensor;
//! let mut t = Tensor::randn(&[3,3], &Default::default()).unwrap();
//! let v = t.view_mut(t.sizes()); // Mutable method while a non-mutable borrow exists (t.sizes())
//! ```
//!
//! We avoid this with;
//! ```rust
//! # use flash_powder::prelude::*;
//! # use flash_powder::Tensor;
//! let mut t = Tensor::randn(&[3,3], &Default::default()).unwrap();
//! let shape = t.shape();
//! let v = t.view_mut(&shape);
//! ```
//!
//! This is the rough equivalent to `torch.Size`.
//!
//! It can deref into `&[usize]`!

#[derive(Default, Clone, Debug)]
pub struct Size(tinyvec::TinyVec<[usize; 8]>);

impl Size {
    pub fn from(v: &[usize]) -> Size {
        Size(v.iter().copied().collect())
    }
}

impl std::ops::Deref for Size {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}
impl PartialEq<&[usize]> for &Size {
    fn eq(&self, other: &&[usize]) -> bool {
        self.0.as_slice() == *other
    }
}
