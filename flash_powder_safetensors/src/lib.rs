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
//!     let options = fp::nn::StateDictLoadOptions::default();
//!     new_linear.load_state_dict(&reader, &options)?;
//!     assert!(new_linear.weight.is_equal(&fp::Tensor::from(&[3.3])?)?);
//!
//!#    Ok(())
//!# }
//! ```
//!
use flash_powder as fp;
use fp::Ten;
use fp::nn;

use anyhow::bail;
use flash_powder::prelude::*;
pub use safetensors;
use safetensors::SafeTensors;

pub mod prelude {
    pub use super::StateDictSafetensor;
}

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
        flash_powder::DType::prevent_cast(()) => unreachable!(),
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

/// Extract a [`fp::Tensor`] from an [`SafeTensors`] by name, without copying the data.
///
/// Tensor is on the cpu and borrowing from the [`SafeTensors`] object.
pub fn safetensor_to_ten<'d>(
    tensors: &'d SafeTensors,
    name: &'_ str,
) -> Result<fp::Ten<'d>, anyhow::Error> {
    if let Ok(tensor_view) = tensors.tensor(name) {
        let dtype = safetensor_dtype_to_flash_powder_dtype(tensor_view.dtype());
        let sizes = tensor_view.shape();
        // Next we need to calculate stride.
        let mut strides = vec![0; sizes.len()];
        let mut current_stride = 1;

        // Iterate backwards from the last dimension to the first
        for i in (0..sizes.len()).rev() {
            strides[i] = current_stride;
            current_stride *= sizes[i];
        }

        let options = fp::tensor::BlobOptionsBytes {
            sizes,
            strides: &strides,
            dtype,
        };
        let data = tensor_view.data();

        // Copy the bytes.
        fp::Ten::from_bytes(data, &options)
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

    pub fn to_state_dict(
        &self,
        options: &flash_powder::factory::ToOptions,
    ) -> Result<nn::StateDict, anyhow::Error> {
        let mut d = nn::StateDict::default();
        for k in self.keys() {
            let ten_view = self
                .ten(&k)
                .ok_or(anyhow::anyhow!("failed to find tensor {k}"))?;
            d.add_data(&k, nn::Data::Buffer(ten_view.to(options)?))?;
        }
        Ok(d)
    }
}

/// Super thin wrapper around [`memmap2::Mmap`].
pub struct MappedFile {
    pub mmap: memmap2::Mmap,
}

impl MappedFile {
    pub fn map<Q>(path: Q) -> Result<MappedFile, anyhow::Error>
    where
        Q: AsRef<std::path::Path>,
    {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(MappedFile { mmap })
    }
    pub fn to_safetensors<'a>(&'a self) -> Result<safetensors::SafeTensors<'a>, anyhow::Error> {
        let tensors = safetensors::SafeTensors::deserialize(&self.mmap)?;
        Ok(tensors)
    }
}

impl<'a, 'd> nn::StateDictAdaptor for SafetensorReader<'a, 'd> {
    fn ten(&self, name: &str) -> Option<Ten<'a>> {
        safetensor_to_ten(self.st, name).ok()
    }

    fn keys(&self) -> std::collections::HashSet<String> {
        self.st.names().iter().map(|v| (*v).to_owned()).collect()
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

pub trait StateDictSafetensor {
    fn serialize_safetensors(&self) -> Result<Vec<u8>, anyhow::Error>;
    fn deserialize_safetensors(
        data: &[u8],
        options: &fp::factory::ToOptions,
    ) -> Result<fp::nn::StateDict, anyhow::Error>;

    fn write_safetensors<Q>(&self, path: Q) -> Result<(), anyhow::Error>
    where
        Q: AsRef<std::path::Path>;

    fn read_safetensors<Q>(
        path: Q,
        options: &fp::factory::ToOptions,
    ) -> Result<fp::nn::StateDict, anyhow::Error>
    where
        Q: AsRef<std::path::Path>;
}
impl StateDictSafetensor for fp::nn::module::StateDict {
    fn write_safetensors<Q>(&self, path: Q) -> Result<(), anyhow::Error>
    where
        Q: AsRef<std::path::Path>,
    {
        let mut data = vec![];
        for (k, v) in self.as_map().iter() {
            data.push((k, SafetensorView::new(v.as_tensor()?)?));
        }
        let p: &std::path::Path = path.as_ref();
        safetensors::tensor::serialize_to_file(data, None, p).map_err(|a| a.into())
    }
    fn read_safetensors<Q>(
        path: Q,
        options: &fp::factory::ToOptions,
    ) -> Result<fp::nn::StateDict, anyhow::Error>
    where
        Q: AsRef<std::path::Path>,
    {
        let mapped = MappedFile::map(&path)?;
        let tensors = mapped.to_safetensors()?;
        let our_safetensor = SafetensorReader::from_safetensors(&tensors);
        our_safetensor.to_state_dict(options)
    }

    fn serialize_safetensors(&self) -> Result<Vec<u8>, anyhow::Error> {
        let mut data = vec![];
        for (k, v) in self.as_map().iter() {
            data.push((k, SafetensorView::new(v.as_tensor()?)?));
        }
        safetensors::tensor::serialize(data, None).map_err(|a| a.into())
    }

    fn deserialize_safetensors(
        data: &[u8],
        options: &fp::factory::ToOptions,
    ) -> Result<fp::nn::StateDict, anyhow::Error> {
        let tensors = safetensors::SafeTensors::deserialize(data)?;
        let our_safetensor = SafetensorReader::from_safetensors(&tensors);
        our_safetensor.to_state_dict(options)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nn::Module;
    #[test]
    fn test_roundtrip() -> Result<(), anyhow::Error> {
        let conv_for_dimensions = fp::nn::Conv2d::new(1, 1, (5, 5), Default::default())?;
        let weight = fp::Tensor::randn(&conv_for_dimensions.weight.shape(), &Default::default())?;
        let bias = fp::Tensor::randn(
            &conv_for_dimensions.bias.as_ref().unwrap().shape(),
            &Default::default(),
        )?;
        let conv = fp::nn::Conv2d {
            weight,
            bias: Some(bias),
            options: Default::default(),
        };

        let adapted = conv.tensors();

        let safetensor_bytes = safetensors::tensor::serialize(
            adapted
                .iter()
                .map(|(k, v)| (k, SafetensorView::new(v).unwrap())),
            None,
        )?;
        assert!(!safetensor_bytes.is_empty());

        // Super cool, now load our serialized data again.
        let tensors = safetensors::SafeTensors::deserialize(&safetensor_bytes)?;
        let reader = SafetensorReader::from_safetensors(&tensors);

        let mut new_conv = fp::nn::Conv2d::new(1, 1, (5, 5), Default::default())?;
        let options = fp::nn::StateDictLoadOptions::default();
        new_conv.load_state_dict(&reader, &options)?;
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
        let weight = fp::Tensor::from([[3.3f32]])?;
        let content = vec![("weight", SafetensorView::new(&weight).unwrap())];
        let safetensor_bytes = safetensors::tensor::serialize(content, None)?;

        // Then deserialize this, normally safetensor_bytes would come from disk!
        let tensors = safetensors::SafeTensors::deserialize(&safetensor_bytes)?;
        let reader = SafetensorReader::from_safetensors(&tensors);

        // Create the destination nn::Module object.
        let mut new_linear = fp::nn::Linear::new_without_bias(1, 1)?;
        // Read into its tensors.
        let options = fp::nn::StateDictLoadOptions::default();
        new_linear.load_state_dict(&reader, &options)?;
        assert!(new_linear.weight.is_equal(&fp::Tensor::from([[3.3f32]])?)?);

        Ok(())
    }

    #[test]
    fn test_statedict() -> Result<(), anyhow::Error> {
        let weight = fp::Tensor::from([[3.3f32]])?;
        let mut d = fp::nn::StateDict::default();
        d.add_data("weight", nn::Data::Parameter(weight))?;
        let path = "/tmp/statedict.safetensor";
        d.write_safetensors(path)?;

        let v = std::fs::read(path)?;
        let serialized = d.serialize_safetensors()?;
        assert_eq!(&v, &serialized);

        let back = flash_powder::nn::StateDict::read_safetensors(path, &Default::default())?;

        let deserialized =
            flash_powder::nn::StateDict::deserialize_safetensors(&serialized, &Default::default())?;

        assert!(
            d.as_map()["weight"]
                .as_tensor()?
                .is_equal(back.as_map()["weight"].as_tensor()?)?
        );

        assert!(
            deserialized.as_map()["weight"]
                .as_tensor()?
                .is_equal(back.as_map()["weight"].as_tensor()?)?
        );

        Ok(())
    }
}
