use torch_stable::headeronly::core::ScalarType;

pub trait ScalarDType {
    fn type_dtype() -> DType;
}
macro_rules! impl_tensor_scalar_dtype_trait {
    ($t:ty, $v:path) => {
        impl ScalarDType for $t {
            fn type_dtype() -> DType {
                $v
            }
        }
    };
}

impl_tensor_scalar_dtype_trait!(f32, DType::F32);
impl_tensor_scalar_dtype_trait!(f64, DType::F64);
// impl_tensor_scalar_trait!(f16, ScalarType::Half);
// https://github.com/pytorch/pytorch/blob/6a357dd272853cb6567bb277da62750013c76b4a/torch/csrc/stable/stableivalue_conversions.h#L114
impl_tensor_scalar_dtype_trait!(u8, DType::U8);
impl_tensor_scalar_dtype_trait!(i8, DType::I8);
impl_tensor_scalar_dtype_trait!(u16, DType::U16);
impl_tensor_scalar_dtype_trait!(i16, DType::I16);
impl_tensor_scalar_dtype_trait!(i32, DType::I32);
impl_tensor_scalar_dtype_trait!(u32, DType::U32);
impl_tensor_scalar_dtype_trait!(i64, DType::I64);
impl_tensor_scalar_dtype_trait!(u64, DType::U64);
impl_tensor_scalar_dtype_trait!(bool, DType::Bool);

/// Simplified ScalarType enum.
///
/// The [`ScalarType`] contains many entries that are not in <https://docs.pytorch.org/docs/2.12/tensor_attributes.html>
/// as well as having the legacy byte/char like naming, by introducing this indirection things are nice and uniform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum DType {
    /// U8
    U8,
    /// I8
    I8,
    /// I16
    I16,
    /// I32
    #[default]
    I32,
    /// I64
    I64,
    /// F16
    F16,
    /// F32
    F32,
    /// F64
    F64,
    /// 32 bit complex with two F16 components.
    Complex32,
    /// 64 bit complex with two F32 components.
    Complex64,
    /// 128 bit complex with two F64 components,
    Complex128,
    /// Boolean
    Bool,
    /// U16
    U16,
    /// U32
    U32,
    /// U64
    U64,
    // Weird ones below
    /// 8-bit floating point, S-E-M 1-5-2
    ///
    /// This is unprintable.
    F8_e5m2,
    /// 8-bit floating point, S-E-M 1-4-3
    ///
    /// This is unprintable.
    F8_e4m3fn,
    /// 8-bit floating point, S-E-M 1-5-2
    ///
    /// This is unprintable.
    F8_e5m2fnuz,
    /// 8-bit floating point, S-E-M 1-4-3
    ///
    /// This is unprintable.
    F8_e4m3fnuz,
    /// 8-bit floating point, S-E-M 0-8-0
    ///
    /// This is unprintable.
    F8_e8m0fnu,
    /// packed 4-bit floating point, S-E-M 1-2-1
    ///
    /// This is unprintable.
    F4_e2m1fn_x2,
    /// BFloat16
    ///
    /// This type is unprintable!
    BF16,

    // This is here to prevent casting the enum to integers, since that MUST go through ScalarType.
    #[doc(hidden)]
    __prevent_cast(()),
}

impl From<DType> for ScalarType {
    fn from(value: DType) -> Self {
        match value {
            DType::U8 => ScalarType::Byte,
            DType::I8 => ScalarType::Char,
            DType::I16 => ScalarType::Short,
            DType::I32 => ScalarType::Int,
            DType::I64 => ScalarType::Long,
            DType::F16 => ScalarType::Half,
            DType::BF16 => ScalarType::BFloat16,
            DType::F32 => ScalarType::Float,
            DType::F64 => ScalarType::Double,
            DType::Complex32 => ScalarType::ComplexHalf,
            DType::Complex64 => ScalarType::ComplexFloat,
            DType::Complex128 => ScalarType::ComplexDouble,
            DType::Bool => ScalarType::Bool,
            DType::U16 => ScalarType::UInt16,
            DType::U32 => ScalarType::UInt32,
            DType::U64 => ScalarType::UInt64,
            DType::F8_e5m2 => ScalarType::Float8_e5m2,
            DType::F8_e4m3fn => ScalarType::Float8_e4m3fn,
            DType::F8_e5m2fnuz => ScalarType::Float8_e5m2fnuz,
            DType::F8_e4m3fnuz => ScalarType::Float8_e4m3fnuz,
            DType::F8_e8m0fnu => ScalarType::Float8_e8m0fnu,
            DType::F4_e2m1fn_x2 => ScalarType::Float4_e2m1fn_x2,
            __prevent_cast => unreachable!(),
        }
    }
}

impl TryFrom<ScalarType> for DType {
    type Error = anyhow::Error;

    fn try_from(value: ScalarType) -> Result<Self, Self::Error> {
        match value {
            ScalarType::Byte => Ok(DType::U8),
            ScalarType::Char => Ok(DType::I8),
            ScalarType::Short => Ok(DType::I16),
            ScalarType::Int => Ok(DType::I32),
            ScalarType::Long => Ok(DType::I64),
            ScalarType::Half => Ok(DType::F16),
            ScalarType::Float => Ok(DType::F32),
            ScalarType::Double => Ok(DType::F64),
            ScalarType::ComplexHalf => Ok(DType::Complex32),
            ScalarType::ComplexFloat => Ok(DType::Complex64),
            ScalarType::ComplexDouble => Ok(DType::Complex128),
            ScalarType::Bool => Ok(DType::Bool),
            ScalarType::UInt16 => Ok(DType::U16),
            ScalarType::UInt32 => Ok(DType::U32),
            ScalarType::UInt64 => Ok(DType::U64),
            ScalarType::Float8_e5m2 => Ok(DType::F8_e5m2),
            ScalarType::Float8_e4m3fn => Ok(DType::F8_e4m3fn),
            ScalarType::Float8_e5m2fnuz => Ok(DType::F8_e5m2fnuz),
            ScalarType::Float8_e4m3fnuz => Ok(DType::F8_e4m3fnuz),
            ScalarType::Float8_e8m0fnu => Ok(DType::F8_e8m0fnu),
            ScalarType::Float4_e2m1fn_x2 => Ok(DType::F4_e2m1fn_x2),
            _ => {
                anyhow::bail!("unimplemented type {value:?}")
            }
        }
    }
}

impl From<DType> for torch_stable::aoti_torch::StableIValue {
    fn from(value: DType) -> Self {
        let as_scalar: ScalarType = value.into();
        as_scalar.into()
    }
}

impl TryFrom<torch_stable::aoti_torch::StableIValue> for DType {
    type Error = anyhow::Error;
    fn try_from(value: torch_stable::aoti_torch::StableIValue) -> Result<Self, Self::Error> {
        let scalar_type = ScalarType::try_from(value.0 as i32)?;
        scalar_type.try_into()
    }
}
