//! This holds functions that pytorch puts into the torch module.
//!
use crate::tensor::{Ten, Tensor, TensorAccess};
use torch_stable::{
    StableTorchResult, aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor,
    unsafe_call_dispatch_bail,
};

/// Select an index in a dimension
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L5898-L5922)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.select.html)
pub fn select<'a, T: TensorAccess>(
    input: &'a T,
    dim: usize,
    index: usize,
) -> StableTorchResult<Ten<'a>> {
    let mut stack: [StableIValue; 3] = [input.get_tensor().into(), dim.into(), index.into()];
    unsafe_call_dispatch_bail!("aten::select", "int", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;
    let marker = std::marker::PhantomData::<&'a ()>;
    Ok(Ten::new(marker, r))
}

/// Concatenates a sequence of tensors along a new dimension.
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L5932)
/// - [pytorch_equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.stack.html)
pub fn stack<T: TensorAccess>(tensors: &[T], dim: usize) -> StableTorchResult<Tensor> {
    let tensor_list: Vec<StableIValue> = tensors.iter().map(|z| z.get_tensor().into()).collect();
    let mut stack: [StableIValue; 2] = [(&tensor_list[..]).into(), dim.into()];
    unsafe_call_dispatch_bail!("aten::stack", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;
    Ok(Tensor::new(r))
}

/// Stack tensors in sequence horizontally (column wise).
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L5898-L5922)
/// - [pytorch_equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.hstack.html)
pub fn hstack<T: TensorAccess>(tensors: &[T]) -> StableTorchResult<Tensor> {
    let tensor_list: Vec<StableIValue> = tensors.iter().map(|z| z.get_tensor().into()).collect();
    let mut stack: [StableIValue; 1] = [(&tensor_list[..]).into()];
    unsafe_call_dispatch_bail!("aten::hstack", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;
    Ok(Tensor::new(r))
}

/// Stack tensors in sequence vertically (row wise).
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/native_functions.yaml#L5954)
/// - [pytorch_equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.vstack.html)
pub fn vstack<T: TensorAccess>(tensors: &[T]) -> StableTorchResult<Tensor> {
    let tensor_list: Vec<StableIValue> = tensors.iter().map(|z| z.get_tensor().into()).collect();
    let mut stack: [StableIValue; 1] = [(&tensor_list[..]).into()];
    unsafe_call_dispatch_bail!("aten::vstack", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;
    Ok(Tensor::new(r))
}
/// Stack tensors in sequence depthwise (along third axis).
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L5898-L5922)
/// - [pytorch_equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.dstack.html)
pub fn dstack<T: TensorAccess>(tensors: &[T]) -> StableTorchResult<Tensor> {
    let tensor_list: Vec<StableIValue> = tensors.iter().map(|z| z.get_tensor().into()).collect();
    let mut stack: [StableIValue; 1] = [(&tensor_list[..]).into()];
    unsafe_call_dispatch_bail!("aten::dstack", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;
    Ok(Tensor::new(r))
}

/// Concatenates the given sequence of tensors in tensors in the given dimension
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L1433)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.cat.html)
pub fn cat<T>(tensors: &[&T], dim: usize) -> StableTorchResult<Tensor>
where
    T: TensorAccess,
{
    let mut stack: [StableIValue; 2] =
        [tensors.iter().map(|z| z.get_tensor()).collect(), dim.into()];
    unsafe_call_dispatch_bail!("aten::cat", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;

    Ok(Tensor::new(r))
}

pub mod cuda {
    /// Return a bool indicating if CUDA is currently available.
    ///
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.is_available.html)
    ///
    /// Not a strict equivalent, this is not exposed through the bindings, instead we try to allocate a tensor on the
    /// cuda device instead.
    pub fn is_available() -> bool {
        use crate::Device;
        use crate::prelude::*;
        let alloced = crate::Tensor::zeros(&[1], &Device::CUDA.into());
        alloced.is_ok()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Tensor;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_torch_select() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([4,4])
            r = torch.select(d, 0, 2);
            c = torch.select(d, 1, 2);
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
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())

        let r = select(&d, 0, 2)?;
        assert_eq!(r.sizes(), &[4]); // #PYTHON list(r.shape)
        assert_eq!(r.f32_ref(&[0])?, &9.0); // #PYTHON r[ 0].item()
        assert_eq!(r.f32_ref(&[1])?, &10.0); // #PYTHON r[ 1].item()
        assert_eq!(r.f32_ref(&[2])?, &11.0); // #PYTHON r[ 2].item()
        assert_eq!(r.f32_ref(&[3])?, &12.0); // #PYTHON r[ 3].item()

        let c = select(&d, 1, 2)?;
        assert_eq!(c.sizes(), &[4]); // #PYTHON list(c.shape)
        assert_eq!(c.i(0)?.as_f32()?, &3.0); // #PYTHON c[ 0].item()
        assert_eq!(c.i(1)?.as_f32()?, &7.0); // #PYTHON c[ 1].item()
        assert_eq!(c.i(2)?.as_f32()?, &11.0); // #PYTHON c[ 2].item()
        assert_eq!(c.i(3)?.as_f32()?, &15.0); // #PYTHON c[ 3].item()

        Ok(())
    }

    #[test]
    fn test_flash_powder_torch_vstack() -> StableTorchResult<()> {
        /*
            #|PYTHON
            a = torch.tensor([1, 2, 3])
            b = torch.tensor([4, 5, 6])
            s = torch.vstack((a,b))
        */
        let a: Tensor = [1, 2, 3].try_into()?;
        let b: Tensor = [4, 5, 6].try_into()?;

        let s = vstack(&[a, b])?;
        assert_eq!(s.sizes(), &[2, 3]); // #PYTHON list(s.shape)
        assert_eq!(s.i32s_ref()?, &[1, 2, 3, 4, 5, 6]); // #PYTHON list(s.view(-1).tolist())
        Ok(())
    }

    #[test]
    fn test_flash_powder_torch_hstack() -> StableTorchResult<()> {
        /*
            #|PYTHON
            a = torch.tensor([1, 2, 3])
            b = torch.tensor([4, 5, 6])
            s = torch.hstack((a,b))
        */
        let a: Tensor = [1, 2, 3].try_into()?;
        let b: Tensor = [4, 5, 6].try_into()?;

        let s = hstack(&[a, b])?;
        assert_eq!(s.sizes(), &[6]); // #PYTHON list(s.shape)
        assert_eq!(s.i32s_ref()?, &[1, 2, 3, 4, 5, 6]); // #PYTHON list(s.view(-1).tolist())
        Ok(())
    }

    #[test]
    fn test_flash_powder_torch_dstack() -> StableTorchResult<()> {
        /*
            #|PYTHON
            a = torch.tensor([1, 2, 3])
            b = torch.tensor([4, 5, 6])
            s = torch.dstack((a,b))
        */
        let a: Tensor = [1, 2, 3].try_into()?;
        let b: Tensor = [4, 5, 6].try_into()?;

        let s = dstack(&[a, b])?;
        assert_eq!(s.sizes(), &[1, 3, 2]); // #PYTHON list(s.shape)
        assert_eq!(s.i32s_ref()?, &[1, 4, 2, 5, 3, 6]); // #PYTHON list(s.view(-1).tolist())
        Ok(())
    }

    #[test]
    fn test_flash_powder_torch_stack() -> StableTorchResult<()> {
        /*
            #|PYTHON
            x = torch.randn(2, 3)
            xx = torch.stack((x,x)) # same as torch.stack((x, x), dim=0)
            xx1 = torch.stack((x, x), dim=1)
            xx2 = torch.stack((x, x), dim=2)
        */
        let x: Tensor = Tensor::randn(&[2, 3], &Default::default())?;

        let xx = stack(&[&x, &x], 0)?;
        assert_eq!(xx.sizes(), &[2, 2, 3]); // #PYTHON list(xx.shape)

        let xx1 = stack(&[&x, &x], 1)?;
        assert_eq!(xx1.sizes(), &[2, 2, 3]); // #PYTHON list(xx1.shape)

        let xx2 = stack(&[&x, &x], 2)?;
        assert_eq!(xx2.sizes(), &[2, 3, 2]); // #PYTHON list(xx2.shape)

        // And there's -1, but our dim is signed atm.
        Ok(())
    }
    #[test]
    fn test_flash_powder_cat() -> StableTorchResult<()> {
        /*
            #|PYTHON
            x = torch.tensor([[1.0, 2.0],[3.0, 4.0]], dtype=torch.float)
        */

        let d = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]])?;
        assert_eq!(d.sizes(), &[2, 2]); // #PYTHON list(x.shape)
        assert_eq!(d.f32s_ref()?, &[1.0f32, 2.0, 3.0, 4.0]); // #PYTHON list(x.view(-1).tolist())

        /*
            #|PYTHON
            a = torch.cat([x,x,x], 0)
        */
        let a = cat(&[&d, &d, &d], 0)?;
        assert_eq!(a.sizes(), &[6, 2]); // #PYTHON list(a.shape)
        assert_eq!(
            a.f32s_ref()?,
            &[1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        ); // #PYTHON list(a.view(-1).tolist())
           /*
               #|PYTHON
               b = torch.cat([x,x,x], 1)
           */
        let b = cat(&[&d, &d, &d], 1)?;
        assert_eq!(b.sizes(), &[2, 6]); // #PYTHON list(b.shape)
        assert_eq!(
            b.f32s_ref()?,
            &[1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        ); // #PYTHON list(b.view(-1).tolist())
        Ok(())
    }

}
