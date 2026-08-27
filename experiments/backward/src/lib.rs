use flash_powder::torch_stable::aoti_torch::StableIValue;
use flash_powder::torch_stable::unsafe_call_dispatch_bail;
use flash_powder::{self as fp, TensorAccess as _};
use flash_powder::{StableTorchResult, Ten, TenMut, Tensor};
use flash_powder::{TensorAccess, prelude::*};
use torch_stable::stable::tensor::Tensor as StableTensor;

trait GradientThings {
    fn requires_grad(self) -> StableTorchResult<Self>
    where
        Self: Sized;

    // https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L41
    // https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.backward.html
    fn backward<A: TensorAccess>(
        &self,
        inputs: &[&A],
        gradient: Option<&A>,
        retain_graph: Option<bool>,
        create_graph: Option<bool>,
    ) -> StableTorchResult<()>;
}
impl GradientThings for Tensor {
    fn requires_grad(self) -> StableTorchResult<Self>
    where
        Self: Sized,
    {
        // https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L83
        let mut stack: [StableIValue; 2] = [self.get_tensor().into(), true.into()];
        unsafe_call_dispatch_bail!("aten::requires_grad_", "", stack.as_mut_slice());

        let r: StableTensor = stack[0].try_into()?;
        Ok(Tensor::new(r))
    }

    fn backward<A: TensorAccess>(
        &self,
        inputs: &[&A],
        gradient: Option<&A>,
        retain_graph: Option<bool>,
        create_graph: Option<bool>,
    ) -> StableTorchResult<()> {
        let create_graph = create_graph.unwrap_or(false);
        // https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml#L41
        let gradient: Option<StableIValue> = gradient.map(|z| z.get_tensor().into());
        let mut stack: [StableIValue; 5] = [
            self.get_tensor().into(),
            inputs.iter().map(|z| z.get_tensor()).collect(),
            (&gradient).into(),
            (&retain_graph).into(),
            create_graph.into(),
        ];
        unsafe_call_dispatch_bail!("aten::_backward", "", stack.as_mut_slice());
        Ok(())
    }
}

/*
from;
# From https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

*/
pub fn main() -> anyhow::Result<()> {
    println!("hello");

    let a: Tensor = [2., 3.].try_into()?;
    let a = a.requires_grad()?;

    let b: Tensor = [6., 4.].try_into()?;
    let b = b.requires_grad()?;

    let c_3: Tensor = 3.0.try_into()?;
    let Q = c_3.mul(&a)?.mul(&a)?.mul(&a)?.sub(&b.mul(&b)?)?;
    let external_grad: Tensor = [1., 1.].try_into()?;

    let _ = Q.backward(&[], Some(&external_grad), None, Some(true))?;

    /*
    [W827 19:47:26.609869767 engine.cpp:1307] Warning: Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak. (function operator())

    Error: dispatch failed (aten::_backward, ) at experiments/backward/src/lib.rs:53 (derivative for aten::_foreach_addcmul is not implemented)

    */

    Ok(())
}
