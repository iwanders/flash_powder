use flash_powder as fp;
use fp::Tensor;
use fp::nn;

use anyhow::bail;
use flash_powder::prelude::*;
pub use safetensors;
use safetensors::SafeTensors;

/// Converter to go from safetensors DType to flash_powder DType
pub fn safetensor_dtype_to_scalar_type(v: safetensors::Dtype) -> fp::DType {
    match v {
        safetensors::Dtype::F16 => fp::DType::F16,
        safetensors::Dtype::F32 => fp::DType::F32,
        safetensors::Dtype::F64 => fp::DType::F64,
        _ => todo!("todo handle {v:?}"),
    }
}

/// Convert a tensor by `name` from `tensors` into a flash powder Tensor.
pub fn safetensor_to_tensor(
    tensors: &SafeTensors,
    name: &str,
) -> Result<fp::Tensor, anyhow::Error> {
    if let Ok(tensor_view) = tensors.tensor(name) {
        // Create a tensor of the correct shape and type
        let mut v = fp::Tensor::zeros(
            tensor_view.shape(),
            &fp::factory::TensorOptions {
                dtype: Some(safetensor_dtype_to_scalar_type(tensor_view.dtype())),
                ..Default::default()
            },
        )?;

        // Copy the bytes.
        v.data_mut()?.copy_from_slice(tensor_view.data());
        Ok(v)
    } else {
        bail!("could not find safetensor {name}")
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SafetensorReader<'a, 'd> {
    st: &'a SafeTensors<'d>,
}
impl<'a, 'd> SafetensorReader<'a, 'd> {
    pub fn from_safetensors(st: &'a SafeTensors<'d>) -> Self {
        Self { st }
    }
}

impl<'a, 'd> nn::StateDictAdaptor for SafetensorReader<'a, 'd> {
    fn tensor(&self, name: &str) -> Option<Tensor> {
        safetensor_to_tensor(self.st, name).ok()
    }
}
impl<'a, 'd> nn::StateDictReader for SafetensorReader<'a, 'd> {
    fn inner(&self) -> &dyn nn::StateDictAdaptor {
        self
    }
}
