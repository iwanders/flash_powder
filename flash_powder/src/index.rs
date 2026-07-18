//! Indexing

use crate::core_methods::{CoreMethods, CoreMethodsMut};
use crate::properties::TensorProperties;
use crate::tensor::{Ten, TenMut, Tensor};
use crate::{StableTorchResult, TensorAccess};
use torch_stable::stable::tensor::Tensor as StableTensor;

// https://docs.pytorch.org/docs/2.11/tensor_view.html
// torch just follows https://numpy.org/doc/stable/user/basics.indexing.html ?
//
// > PyTorch follows Numpy behaviors that basic indexing returns views, while advanced indexing returns a copy.
// > Assignment via either basic or advanced indexing is in-place.
// > See more examples in Numpy indexing documentation.
//
// :<
//
// Advanced indexing; https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
// Basically, if indexing is non tuple, like ndarray with bool/integer, or a tuple with sequence of int/bool.
// So basically whenever a select happens?
//
//
// https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/TensorIndexing.h#L88-L112
// This definitely has Slice(start, stop, step)
//
// Python slices have a step though :| Rust's ranges do not...
// Maybe we do something like (0..12,3) for 0:12:3 -> [0, 3, 6, 9]?
// We can probably implement Into TensorIndexOptions for both Range and (Range, usize) or something?
//
// Should we use isize or usize... usize feels more natural... but isize can walk backwards.

// Do we even need the step for indexing? Works in native lists.
// a = list(range(16))
// a[0:12:3] ->[0, 3, 6, 9]
//
// also works in numpy;
//
// np.array(list(range(16)))[0:12:3]
// array([0, 3, 6, 9])

// That https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/TensorIndexing.h#L438 looks very complex :/
//

/*

Tricky:
    - Indexing with tensors with index.Tensor always returns a copy, but the current indexing system returns a view.
      We can't reconcile this without an extra method, or indexing overload or something. For now we can use index_tensor,
*/

#[derive(Clone)]
pub enum TensorIndexOptions<'a> {
    // Can we even do this? this indexing method seems to return a copy instead of a view.
    Tensor(&'a StableTensor),
    Index(isize),
    Range(std::ops::Range<isize>),
    RangeFull,
    // Can we even do this?
    RangeWithStride {
        range: std::ops::Range<usize>,
        stride: isize,
    },
}
impl<'a> std::fmt::Debug for TensorIndexOptions<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(_arg0) => f.debug_tuple("Tensor").finish(), //.field(arg0)
            Self::Index(arg0) => f.debug_tuple("Index").field(arg0).finish(),
            Self::Range(arg0) => f.debug_tuple("Range").field(arg0).finish(),
            Self::RangeFull => write!(f, "RangeFull"),
            Self::RangeWithStride { range, stride } => f
                .debug_struct("RangeWithStride")
                .field("range", range)
                .field("stride", stride)
                .finish(),
        }
    }
}

impl<'a> From<isize> for TensorIndexOptions<'a> {
    fn from(val: isize) -> Self {
        TensorIndexOptions::Index(val)
    }
}
impl<'a> From<std::ops::Range<isize>> for TensorIndexOptions<'a> {
    fn from(val: std::ops::Range<isize>) -> Self {
        TensorIndexOptions::Range(val.clone())
    }
}
impl<'a> From<std::ops::RangeFull> for TensorIndexOptions<'a> {
    fn from(val: std::ops::RangeFull) -> Self {
        let _ = val;
        TensorIndexOptions::RangeFull
    }
}
// impl<'a> From<&'a Ten<'_>> for TensorIndexOptions<'a> {
//     fn from(val: &'a Ten<'_>) -> Self {
//         TensorIndexOptions::Tensor(val.get_tensor())
//     }
// }

trait TensorIndexWorker: CoreMethods {
    fn do_the_real_indexing<'a, 'b>(
        &'b self,
        index: &[&TensorIndexOptions<'a>],
    ) -> StableTorchResult<Ten<'b>> {
        // Make a view into the tensor, we'll be updating this as we go through the indexing operations.
        let shape = self.shape();
        let mut current = self.ten()?;
        let mut current_dim = 0;
        let mut source_dim = 0;
        const DEBUG: bool = false;
        for index_op_conv in index.iter() {
            if DEBUG {
                println!(
                    "current_dim {current_dim:?},  source_dim {source_dim:?}: Index op {index_op_conv:?}"
                );
            }
            match index_op_conv {
                TensorIndexOptions::Tensor(_tensor) => todo!(),
                TensorIndexOptions::Index(index) => {
                    if DEBUG {
                        println!("Before select shape: {:?}", current.shape());
                    }
                    current = current.select(current_dim, *index as usize)?;
                    // current = current.narrow(current_dim, *index, 1)?.into_squeeze()?;
                    if DEBUG {
                        println!("After select shape: {:?}", current.shape());
                    }
                    //current_dim += 1;
                    source_dim += 1;
                }
                TensorIndexOptions::Range(range) => {
                    let length = if range.start < 0 {
                        (shape[current_dim] as isize + range.start) as usize + 1
                    } else {
                        range.len()
                    };
                    current = current.narrow(current_dim, range.start, length)?;
                    current_dim += 1;
                    source_dim += 1;
                }
                TensorIndexOptions::RangeFull => {
                    current = current.narrow(current_dim, 0, shape[source_dim])?;
                    current_dim += 1;
                    source_dim += 1;
                }
                TensorIndexOptions::RangeWithStride {
                    range: _range,
                    stride: _stride,
                } => todo!(),
            }
        }
        Ok(current)
    }
}

impl TensorIndexWorker for Tensor {}
impl TensorIndexWorker for Ten<'_> {}
impl TensorIndexWorker for TenMut<'_> {}

pub trait IndexSpec<T> {
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>>;
}
pub trait TensorIndex: TensorAccess + TensorProperties + CoreMethods + Sized {
    fn i<'a, I: IndexSpec<Self>>(&'a self, index: I) -> StableTorchResult<Ten<'a>> {
        index.do_index(self)
    }
}
impl TensorIndex for Tensor {}
impl TensorIndex for Ten<'_> {}
impl TensorIndex for TenMut<'_> {}

impl<'a, A: Clone, T: TensorIndexWorker> IndexSpec<T> for A
where
    A: Into<TensorIndexOptions<'a>>,
{
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>> {
        let first: TensorIndexOptions<'_> = self.clone().into();
        tensor.do_the_real_indexing(&[&first])
    }
}

impl<'a, A: Clone, B: Clone, T: TensorIndexWorker> IndexSpec<T> for (A, B)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
{
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        tensor.do_the_real_indexing(&[&first, &second])
    }
}
impl<'a, A: Clone, B: Clone, C: Clone, T: TensorIndexWorker> IndexSpec<T> for (A, B, C)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
    C: Into<TensorIndexOptions<'a>>,
{
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        let third: TensorIndexOptions<'_> = self.2.clone().into();
        tensor.do_the_real_indexing(&[&first, &second, &third])
    }
}

impl<'a, A: Clone, B: Clone, C: Clone, D: Clone, T: TensorIndexWorker> IndexSpec<T> for (A, B, C, D)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
    C: Into<TensorIndexOptions<'a>>,
    D: Into<TensorIndexOptions<'a>>,
{
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        let third: TensorIndexOptions<'_> = self.2.clone().into();
        let four: TensorIndexOptions<'_> = self.3.clone().into();
        tensor.do_the_real_indexing(&[&first, &second, &third, &four])
    }
}

impl<'a, A: Clone, B: Clone, C: Clone, D: Clone, E: Clone, T: TensorIndexWorker> IndexSpec<T>
    for (A, B, C, D, E)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
    C: Into<TensorIndexOptions<'a>>,
    D: Into<TensorIndexOptions<'a>>,
    E: Into<TensorIndexOptions<'a>>,
{
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        let third: TensorIndexOptions<'_> = self.2.clone().into();
        let four: TensorIndexOptions<'_> = self.3.clone().into();
        let five: TensorIndexOptions<'_> = self.4.clone().into();
        tensor.do_the_real_indexing(&[&first, &second, &third, &four, &five])
    }
}

// ------------------------------------------------------------------------------------------
// and the mut flavours.
trait TensorIndexWorkerMut: CoreMethodsMut + CoreMethods {
    fn do_the_real_indexing_mut<'a, 'b>(
        &'b mut self,
        index: &[&TensorIndexOptions<'a>],
    ) -> StableTorchResult<TenMut<'b>> {
        // Make a view into the tensor, we'll be updating this as we go through the indexing operations.
        let shape = self.shape();
        let mut current = self.ten_mut()?;
        let mut current_dim = 0;
        let mut source_dim = 0;
        const DEBUG: bool = false;
        for index_op_conv in index.iter() {
            if DEBUG {
                println!(
                    "current_dim {current_dim:?},  source_dim {source_dim:?}: Index op {index_op_conv:?}"
                );
            }
            match index_op_conv {
                TensorIndexOptions::Tensor(_tensor) => todo!(),
                TensorIndexOptions::Index(index) => {
                    if DEBUG {
                        println!("Before select shape: {:?}", current.shape());
                    }
                    current = current.into_select_mut(current_dim, *index as usize)?;
                    // current = current.narrow(current_dim, *index, 1)?.into_squeeze()?;
                    if DEBUG {
                        println!("After select shape: {:?}", current.shape());
                    }
                    //current_dim += 1;
                    source_dim += 1;
                }
                TensorIndexOptions::Range(range) => {
                    let length = if range.start < 0 {
                        (shape[current_dim] as isize + range.start) as usize + 1
                    } else {
                        range.len()
                    };
                    current = current.into_narrow_mut(current_dim, range.start, length)?;
                    current_dim += 1;
                    source_dim += 1;
                }
                TensorIndexOptions::RangeFull => {
                    current = current.into_narrow_mut(current_dim, 0, shape[source_dim])?;
                    current_dim += 1;
                    source_dim += 1;
                }
                TensorIndexOptions::RangeWithStride {
                    range: _range,
                    stride: _stride,
                } => todo!(),
            }
        }
        Ok(current)
    }
}

impl TensorIndexWorkerMut for Tensor {}
impl TensorIndexWorkerMut for TenMut<'_> {}

pub trait IndexSpecMut<T> {
    fn do_index_mut<'b>(&self, tensor: &'b mut T) -> StableTorchResult<TenMut<'b>>;
}
pub trait TensorIndexMut: TensorAccess + TensorProperties + CoreMethodsMut + Sized {
    fn i_mut<'a, I: IndexSpecMut<Self>>(&'a mut self, index: I) -> StableTorchResult<TenMut<'a>> {
        index.do_index_mut(self)
    }
}
impl TensorIndexMut for Tensor {}
impl TensorIndexMut for TenMut<'_> {}

impl<'a, A: Clone, T: TensorIndexWorkerMut> IndexSpecMut<T> for A
where
    A: Into<TensorIndexOptions<'a>>,
{
    fn do_index_mut<'b>(&self, tensor: &'b mut T) -> StableTorchResult<TenMut<'b>> {
        let first: TensorIndexOptions<'_> = self.clone().into();
        tensor.do_the_real_indexing_mut(&[&first])
    }
}

impl<'a, A: Clone, B: Clone, T: TensorIndexWorkerMut> IndexSpecMut<T> for (A, B)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
{
    fn do_index_mut<'b>(&self, tensor: &'b mut T) -> StableTorchResult<TenMut<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        tensor.do_the_real_indexing_mut(&[&first, &second])
    }
}
impl<'a, A: Clone, B: Clone, C: Clone, T: TensorIndexWorkerMut> IndexSpecMut<T> for (A, B, C)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
    C: Into<TensorIndexOptions<'a>>,
{
    fn do_index_mut<'b>(&self, tensor: &'b mut T) -> StableTorchResult<TenMut<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        let third: TensorIndexOptions<'_> = self.2.clone().into();
        tensor.do_the_real_indexing_mut(&[&first, &second, &third])
    }
}

impl<'a, A: Clone, B: Clone, C: Clone, D: Clone, T: TensorIndexWorkerMut> IndexSpecMut<T>
    for (A, B, C, D)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
    C: Into<TensorIndexOptions<'a>>,
    D: Into<TensorIndexOptions<'a>>,
{
    fn do_index_mut<'b>(&self, tensor: &'b mut T) -> StableTorchResult<TenMut<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        let third: TensorIndexOptions<'_> = self.2.clone().into();
        let four: TensorIndexOptions<'_> = self.3.clone().into();
        tensor.do_the_real_indexing_mut(&[&first, &second, &third, &four])
    }
}

impl<'a, A: Clone, B: Clone, C: Clone, D: Clone, E: Clone, T: TensorIndexWorkerMut> IndexSpecMut<T>
    for (A, B, C, D, E)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
    C: Into<TensorIndexOptions<'a>>,
    D: Into<TensorIndexOptions<'a>>,
    E: Into<TensorIndexOptions<'a>>,
{
    fn do_index_mut<'b>(&self, tensor: &'b mut T) -> StableTorchResult<TenMut<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        let third: TensorIndexOptions<'_> = self.2.clone().into();
        let four: TensorIndexOptions<'_> = self.3.clone().into();
        let five: TensorIndexOptions<'_> = self.4.clone().into();
        tensor.do_the_real_indexing_mut(&[&first, &second, &third, &four, &five])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_indexing() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([ 4,4])
        */

        let d = Tensor::from([
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

        let z = d.i((1..3, 0..1))?;
        /*
            #|PYTHON
            z = d[1:3, 0:1]
        */

        assert_eq!(z.is_contiguous(), false);
        assert_eq!(z.sizes(), &[2, 1]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 4); // #PYTHON  (z.stride(0))
        assert_eq!(z.stride(1), 1); // #PYTHON  (z.stride(1))
        assert_eq!(z.i((0, 0))?.as_f32()?, &5.0); // #PYTHON z[0,0].item()
        assert_eq!(z.i((1, 0))?.as_f32()?, &9.0); // #PYTHON z[1,0].item()

        // Ah yes, now we need a tuple...
        //
        let z = d.i((1, 0..3))?;
        /*
            #|PYTHON
            z = d[1, 0:3]
        */

        assert_eq!(z.sizes(), &[3]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 1); // #PYTHON  (z.stride(0))
        assert_eq!(z.f32_ref(&[0])?, &5.0); // #PYTHON z[0].item()
        assert_eq!(z.f32_ref(&[1])?, &6.0); // #PYTHON z[1].item()
        assert_eq!(z.f32_ref(&[2])?, &7.0); // #PYTHON z[2].item()

        let z = d.i((0..3, 1))?;
        /*
            #|PYTHON
            z = d[0:3, 1]
        */

        assert_eq!(z.sizes(), &[3]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 4); // #PYTHON  (z.stride(0))
        assert_eq!(z.i(0)?.as_f32()?, &2.0); // #PYTHON z[0].item()
        assert_eq!(z.i(1)?.as_f32()?, &6.0); // #PYTHON z[1].item()
        assert_eq!(z.i(2)?.as_f32()?, &10.0); // #PYTHON z[2].item()

        let z = d.i((0..3, 1))?;
        /*
            #|PYTHON
            z = d[0:3, 1]
        */

        //println!("z: {z:?}");
        assert_eq!(z.sizes(), &[3]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 4); // #PYTHON  (z.stride(0))
        assert_eq!(z.i(0)?.as_f32()?, &2.0); // #PYTHON z[0].item()
        assert_eq!(z.i(1)?.as_f32()?, &6.0); // #PYTHON z[1].item()
        assert_eq!(z.i(2)?.as_f32()?, &10.0); // #PYTHON z[2].item()

        let z = d.i((-3isize..3, -3isize..3))?;
        /*
            #|PYTHON
            z = d[-3:3, -3:3]
        */

        assert_eq!(z.sizes(), &[2, 2]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 4); // #PYTHON  (z.stride(0))
        assert_eq!(z.i((0, 0))?.as_f32()?, &6.0); // #PYTHON z[0,0].item()
        assert_eq!(z.i((1, 0))?.as_f32()?, &10.0); // #PYTHON z[1,0].item()
        assert_eq!(z.i((1, 1))?.as_f32()?, &11.0); // #PYTHON z[1,1].item()

        Ok(())
    }

    #[test]
    fn test_flash_powder_indexing_mut() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([ 4,4])
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

        /*
            #|PYTHON
            z = d[1:3]
        */

        let mut z = d.i_mut(1..3)?;
        assert_eq!(z.sizes(), &[2, 4]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 4); // #PYTHON  (z.stride(0))
        assert_eq!(z.f32_ref(&[0, 2])?, &7.0); // #PYTHON z[0, 2].item()

        assert_eq!(z.f32_ref(&[1, 2])?, &11.0); // #PYTHON z[1, 2].item()

        // and then overwrite it.
        /*
            #|PYTHON
            z[1,2]  = 100.0
        */

        *(z.f32_mut(&[1, 2])?) = 100.0;
        assert_eq!(z.f32_ref(&[1, 2])?, &100.0); // #PYTHON z[1, 2].item()
        assert_eq!(d.f32_ref(&[2, 2])?, &100.0); // #PYTHON d[2, 2].item()

        // borrow on borrow.
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([ 4,4])
            z = d[1:3]
            y = z[0:1]
            y[0,0] = 120.0
        */
        let mut z = d.i_mut(1..3)?;
        let mut y = z.i_mut(0..1)?;
        assert_eq!(y.sizes(), &[1, 4]); // #PYTHON list(y.shape)
        println!("v: {}", y.f32_mut(&[0, 0])?);
        *(y.f32_mut(&[0, 0])?) = 120.0;
        assert_eq!(d.f32_ref(&[1, 0])?, &120.0); // #PYTHON d[1, 0].item()

        Ok(())
    }

    #[test]
    fn test_flash_powder_indexing_non_contiguous_scalar() -> StableTorchResult<()> {
        let d = Tensor::from([
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ])?;
        let x = d.permute(&[1, 0])?;
        assert_eq!(x.is_contiguous(), false);

        let topleft = x.i((0, 0))?;
        assert_eq!(topleft.is_contiguous(), true);
        assert_eq!(topleft.dim(), 0);

        println!("storage offset: {:?}", topleft.storage_offset());

        let offbyone = x.i((0, 1))?;
        assert_eq!(offbyone.is_contiguous(), true);
        assert_eq!(offbyone.dim(), 0);
        println!("storage offset: {:?}", offbyone.storage_offset());

        let offbyone = x.i((2, 3))?;
        assert_eq!(offbyone.is_contiguous(), true);
        assert_eq!(offbyone.dim(), 0);
        println!("storage offset: {:?}", offbyone.storage_offset());
        Ok(())
    }

    #[test]
    fn test_flash_powder_indexing_why_no_dimension_drop() -> StableTorchResult<()> {
        // Simpler case of below is the actual problem.
        /*
            #|PYTHON
            c = torch.zeros((2, 896))

        */
        let d = Tensor::zeros(&[2, 896], &Default::default())?;
        assert_eq!(d.sizes(), &[2, 896]); // #PYTHON list(c.shape)

        /*
            #|PYTHON
            u = c[0, :]

        */
        let u = d.i((0, ..))?;
        assert_eq!(u.sizes(), &[896]); // #PYTHON list(u.shape)

        //  if r: [2, 896, 1664], why doesn't r.i((0, .., ..)) make that [1, 896, 1664]?
        /*
            #|PYTHON
            d = torch.zeros((2, 896, 1664))

        */
        let d = Tensor::zeros(&[2, 896, 1664], &Default::default())?;
        assert_eq!(d.sizes(), &[2, 896, 1664]); // #PYTHON list(d.shape)

        /*
            #|PYTHON
            v= d[0, :, :]

        */
        let v = d.i((0, .., ..))?;
        assert_eq!(v.sizes(), &[896, 1664]); // #PYTHON list(v.shape)

        Ok(())
    }
}
