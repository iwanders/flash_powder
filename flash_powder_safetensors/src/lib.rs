//! Helper tooling to read tensors from [safetensors].
//!
//! Core functionality provided by [`SafetensorReader`].
//!
//! Currently not zero copy, tensors or owning and on the cpu.
//!
//! Example usage:
//!
//! ```rust
//!# fn test_minimal() -> Result<(), anyhow::Error> {
//!# use flash_powder_safetensors::*;
//!# use flash_powder as fp;
//!# use fp::{nn, nn::Module};
//!# use fp::prelude::*;
//!     // We create some dummy safetensor data here:
//!     let weight = fp::Tensor::from(&[3.3])?;
//!     let content = vec![("weight", SafetensorView::new(&weight).unwrap())];
//!     let safetensor_bytes = safetensors::tensor::serialize(content, None)?;
//!
//!     // Then deserialize this, normally safetensor_bytes would come from disk!
//!     let tensors = safetensors::SafeTensors::deserialize(&safetensor_bytes)?;
//!     let reader = SafetensorReader::from_safetensors(&tensors);
//!
//!     // Create the destination nn::Module object.
//!     let mut new_linear = fp::nn::Linear::new_without_bias(1, 1)?;
//!     // Read into its tensors.
//!     new_linear.load_state_dict(&reader)?;
//!     assert!(new_linear.weight.is_equal(&fp::Tensor::from(&[3.3])?)?);
//!
//!#    Ok(())
//!# }
//! ```
//!

// Todo, use a borrowed blob; https://github.com/pytorch/pytorch/blob/6a641f6777594fcd2f34ea32f7ee2c0cdaa55776/torch/csrc/stable/ops.h#L680-L725

use flash_powder as fp;
use fp::Tensor;
use fp::nn;

use anyhow::bail;
use flash_powder::prelude::*;
pub use safetensors;
use safetensors::SafeTensors;

/// Convert [`safetensors::Dtype`] to [`fp::DType`].
pub fn safetensor_dtype_to_flash_powder_dtype(v: safetensors::Dtype) -> fp::DType {
    match v {
        safetensors::Dtype::F16 => fp::DType::F16,
        safetensors::Dtype::F32 => fp::DType::F32,
        safetensors::Dtype::F64 => fp::DType::F64,
        safetensors::Dtype::BOOL => fp::DType::Bool,
        safetensors::Dtype::F4 => fp::DType::F4_e2m1fn_x2,
        safetensors::Dtype::F6_E2M3 => todo!(), // I don't know what to do with this :/
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

/// Convert [`fp::DType`] to [`safetensors::Dtype`].
pub fn flash_powder_dtype_to_safetensor_dtype(v: fp::DType) -> safetensors::Dtype {
    match v {
        fp::DType::F16 => safetensors::Dtype::F16,
        fp::DType::F32 => safetensors::Dtype::F32,
        fp::DType::F64 => safetensors::Dtype::F64,
        flash_powder::DType::U8 => safetensors::Dtype::U8,
        flash_powder::DType::I8 => safetensors::Dtype::I8,
        flash_powder::DType::I16 => safetensors::Dtype::I16,
        flash_powder::DType::I32 => safetensors::Dtype::I32,
        flash_powder::DType::I64 => safetensors::Dtype::I64,
        flash_powder::DType::Complex32 => todo!(),
        flash_powder::DType::Complex64 => safetensors::Dtype::C64,
        flash_powder::DType::Complex128 => todo!(),
        flash_powder::DType::Bool => safetensors::Dtype::BOOL,
        flash_powder::DType::U16 => safetensors::Dtype::U16,
        flash_powder::DType::U32 => safetensors::Dtype::U32,
        flash_powder::DType::U64 => safetensors::Dtype::U64,
        flash_powder::DType::F8_e5m2 => safetensors::Dtype::F8_E5M2,
        flash_powder::DType::F8_e4m3fn => todo!(),
        flash_powder::DType::F8_e5m2fnuz => safetensors::Dtype::F8_E5M2FNUZ,
        flash_powder::DType::F8_e4m3fnuz => safetensors::Dtype::F8_E4M3FNUZ,
        flash_powder::DType::F8_e8m0fnu => safetensors::Dtype::F8_E8M0,
        flash_powder::DType::F4_e2m1fn_x2 => todo!(),
        flash_powder::DType::BF16 => todo!(),
    }
}

/// Extract a [`fp::Tensor`] from an [`SafeTensors`] by name.
///
/// Tensor is on the cpu and owning.
pub fn safetensor_to_tensor(
    tensors: &SafeTensors,
    name: &str,
) -> Result<fp::Tensor, anyhow::Error> {
    if let Ok(tensor_view) = tensors.tensor(name) {
        // Create a tensor of the correct shape and type
        let mut v = fp::Tensor::zeros(
            tensor_view.shape(),
            &fp::factory::TensorOptions {
                dtype: Some(safetensor_dtype_to_flash_powder_dtype(tensor_view.dtype())),
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

/// Adaptor struct that implements [`flash_powder::nn::StateDictReader`].
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

/// Adaptor to provide [`safetensors::tensor::View`] for Tensor objects.
pub struct SafetensorView<'a, T: fp::core_methods::CoreMethods + fp::data::DataRef> {
    t: &'a T,
}
impl<'a, T: fp::core_methods::CoreMethods + fp::data::DataRef> SafetensorView<'a, T> {
    pub fn new(t: &'a T) -> Result<Self, anyhow::Error> {
        // Verify that the accessor doesn't, if it works here, it ought to work later.
        let _ = t.data()?;

        Ok(Self { t })
    }
}
impl<'a, T: fp::core_methods::CoreMethods + fp::data::DataRef> safetensors::tensor::View
    for SafetensorView<'a, T>
{
    fn dtype(&self) -> safetensors::Dtype {
        flash_powder_dtype_to_safetensor_dtype(self.t.dtype())
    }

    fn shape(&self) -> &[usize] {
        self.t.sizes()
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        self.t.data().unwrap().into()
    }

    fn data_len(&self) -> usize {
        self.t.data().unwrap().len()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nn::Module;
    #[test]
    fn test_roundtrip() -> Result<(), anyhow::Error> {
        let weight = fp::Tensor::randn(&[5, 5], &Default::default())?;
        let bias = fp::Tensor::randn(&[5, 5], &Default::default())?;
        let conv = fp::nn::Conv2d {
            weight,
            bias: Some(bias),
            options: Default::default(),
        };

        let adapted = conv.tensors();

        let safetensor_bytes = safetensors::tensor::serialize(
            adapted
                .iter()
                .map(|(k, v)| (k, SafetensorView::new(*v).unwrap())),
            None,
        )?;
        assert!(!safetensor_bytes.is_empty());

        // Super cool, now load our serialized data again.
        let tensors = safetensors::SafeTensors::deserialize(&safetensor_bytes)?;
        let reader = SafetensorReader::from_safetensors(&tensors);

        let mut new_conv = fp::nn::Conv2d::new(3, 3, (3, 3), Default::default())?;
        new_conv.load_state_dict(&reader)?;
        assert!(conv.weight.is_equal(&new_conv.weight)?);
        assert!(
            conv.bias
                .as_ref()
                .unwrap()
                .is_equal(new_conv.bias.as_ref().unwrap())?
        );

        Ok(())
    }
    #[test]
    fn test_minimal() -> Result<(), anyhow::Error> {
        // We create some dummy safetensor data here:
        let weight = fp::Tensor::from(&[3.3])?;
        let content = vec![("weight", SafetensorView::new(&weight).unwrap())];
        let safetensor_bytes = safetensors::tensor::serialize(content, None)?;

        // Then deserialize this, normally safetensor_bytes would come from disk!
        let tensors = safetensors::SafeTensors::deserialize(&safetensor_bytes)?;
        let reader = SafetensorReader::from_safetensors(&tensors);

        // Create the destination nn::Module object.
        let mut new_linear = fp::nn::Linear::new_without_bias(1, 1)?;
        // Read into its tensors.
        new_linear.load_state_dict(&reader)?;
        assert!(new_linear.weight.is_equal(&fp::Tensor::from(&[3.3])?)?);

        Ok(())
    }
}
