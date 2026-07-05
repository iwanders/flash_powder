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
        safetensors::Dtype::BOOL => fp::DType::Bool,
        safetensors::Dtype::F4 => fp::DType::F4_e2m1fn_x2,
        safetensors::Dtype::F6_E2M3 => todo!(),
        safetensors::Dtype::F6_E3M2 => todo!(),
        safetensors::Dtype::U8 => fp::DType::U8,
        safetensors::Dtype::I8 => fp::DType::I8,
        safetensors::Dtype::F8_E5M2 => fp::DType::F8_e5m2,
        safetensors::Dtype::F8_E4M3 => fp::DType::F8_e4m3fn,
        safetensors::Dtype::F8_E8M0 => fp::DType::F8_e8m0fnu,
        safetensors::Dtype::F8_E4M3FNUZ => fp::DType::F8_e4m3fnuz,
        safetensors::Dtype::F8_E5M2FNUZ => fp::DType::F8_e5m2fnuz,
        safetensors::Dtype::I16 => fp::DType::I16,
        safetensors::Dtype::U16 => fp::DType::U16,
        safetensors::Dtype::BF16 => fp::DType::BF16,
        safetensors::Dtype::I32 => fp::DType::I32,
        safetensors::Dtype::U32 => fp::DType::U32,
        safetensors::Dtype::C64 => fp::DType::Complex64,
        safetensors::Dtype::I64 => fp::DType::I64,
        safetensors::Dtype::U64 => fp::DType::U64,
        _ => todo!(),
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
