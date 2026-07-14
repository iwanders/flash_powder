//! Deprecated contrib, predates `flash_powder`, use that instead.
//!
// This holds things that aren't actually part of the upstream stable API, but make sense.
// They're implemented as a trait, to make it clear it is opt in.

// Random notes:
// https://github.com/pytorch/pytorch/issues/34646
// https://github.com/pytorch/pytorch/issues/107112
// https://docs.pytorch.org/docs/stable/dlpack.html#torch.utils.dlpack.from_dlpack
//
// How do we dispatch?
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md
// https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough
// https://github.com/pytorch/pytorch/wiki/Codegen-and-Structured-Kernels
// https://blog.ezyang.com/2019/05/pytorch-internals/
//
// let mut stack: [StableIValue; 2] = [self.into(), other.into()];
// unsafe_call_dispatch_bail!("aten::add", "Tensor", stack.as_mut_slice()); // does something, mostly complain about scalars
// stack[0].try_into()
//
use anyhow::bail;

use crate::aoti_torch::AtenTensorHandle;
use crate::headeronly::core::ScalarType;
use crate::stable::ops::EmtpyOptions;
use crate::stable::tensor::Tensor;
use crate::{StableTorchResult, unsafe_call_bail, unsafe_call_panic};
use crate::{aoti_torch::*, unsafe_call_dispatch_bail};
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

pub trait TensorFromScalar {
    fn from_f32(value: f32) -> StableTorchResult<Tensor>;
    fn from_f64(value: f64) -> StableTorchResult<Tensor>;
}

impl TensorFromScalar for Tensor {
    fn from_f32(value: f32) -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_scalar_to_tensor_float32(value, &mut handle_res));
        Ok(Tensor::from_handle(handle_res))
    }
    fn from_f64(value: f64) -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_scalar_to_tensor_float64(value, &mut handle_res));
        Ok(Tensor::from_handle(handle_res))
    }
}

pub trait TensorToScalar {
    fn to_f32(&self) -> StableTorchResult<f32>;
    fn to_f64(&self) -> StableTorchResult<f64>;
}
impl TensorToScalar for Tensor {
    fn to_f32(&self) -> StableTorchResult<f32> {
        let mut sum_result: f32 = 0.0;
        unsafe_call_bail!(aoti_torch_item_float32(self.get(), &mut sum_result));
        Ok(sum_result)
    }

    fn to_f64(&self) -> StableTorchResult<f64> {
        let mut sum_result: f64 = 0.0;
        unsafe_call_bail!(aoti_torch_item_float64(self.get(), &mut sum_result));
        Ok(sum_result)
    }
}

pub trait Math {
    fn add(&self, other: &Tensor) -> StableTorchResult<Tensor>;
}
impl Math for Tensor {
    fn add(&self, other: &Tensor) -> StableTorchResult<Tensor> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();

        if self.is_cpu() && other.is_cpu() {
            unsafe_call_bail!(aoti_torch_cpu_add_Tensor(
                self.get(),
                other.get(),
                1.0,
                &mut handle_res
            ));
        } else if self.is_cuda() && other.is_cuda() {
            #[cfg(feature = "cuda")]
            unsafe_call_bail!(aoti_torch_cuda_add_Tensor(
                self.get(),
                other.get(),
                1.0,
                &mut handle_res
            ));
        }
        Ok(Self::from_handle(handle_res))
    }
}

pub trait DataManipulation {
    fn data_ref(&self) -> StableTorchResult<&[u8]>;
    fn data_mut(&mut self) -> StableTorchResult<&mut [u8]>;

    fn f32_ref(&self) -> StableTorchResult<&[f32]>;
    fn f32_mut(&mut self) -> StableTorchResult<&mut [f32]>;

    fn t_ref<T: IntoBytes + TryFromBytes + Immutable>(&self) -> StableTorchResult<&[T]>;
    fn t_mut<T: IntoBytes + TryFromBytes + Immutable>(&mut self) -> StableTorchResult<&mut [T]>;
}

impl DataManipulation for Tensor {
    fn data_ref(&self) -> StableTorchResult<&[u8]> {
        let element_size = self.element_size();
        let elements = self.numel();
        let data_ptr = self.const_data_ptr();
        if self.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    fn data_mut(&mut self) -> StableTorchResult<&mut [u8]> {
        let element_size = self.element_size();
        let elements = self.numel();
        let data_ptr = self.mutable_data_ptr();
        if self.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts_mut(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    fn f32_ref(&self) -> StableTorchResult<&[f32]> {
        let byte_ref = self.data_ref()?;
        use zerocopy::TryFromBytes;
        match <[f32]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn f32_mut(&mut self) -> StableTorchResult<&mut [f32]> {
        let byte_ref = self.data_mut()?;
        match <[f32]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn t_ref<T: IntoBytes + TryFromBytes + Immutable>(&self) -> StableTorchResult<&[T]> {
        let byte_ref = self.data_ref()?;
        match <[T]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn t_mut<T: IntoBytes + TryFromBytes + Immutable>(&mut self) -> StableTorchResult<&mut [T]> {
        let byte_ref = self.data_mut()?;
        match <[T]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
}

pub trait Creation {
    fn zeros(dimensions: &[usize], options: &EmtpyOptions) -> StableTorchResult<Tensor>;
}
impl Creation for Tensor {
    fn zeros(dimensions: &[usize], options: &EmtpyOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 6] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
            (&options.memory_format).into(),
        ];
        unsafe_call_dispatch_bail!("aten::empty", "memory_format", stack.as_mut_slice());
        let r: Tensor = stack[0].try_into()?;

        unsafe_call_bail!(aoti_torch_zero_(r.get()));

        Ok(r)
        // */
        /*
        let mut stack: [StableIValue; 5] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
        ];
        unsafe_call_dispatch_bail!("aten::zeros", "", stack.as_mut_slice());

        Ok(stack[0].try_into()?)*/
    }
}

pub trait Arange {
    fn arange_u64(start: u64, end: u64, step: u64) -> StableTorchResult<Tensor>;
}
// Should we just pull in num traits here?
impl Arange for Tensor {
    fn arange_u64(start: u64, end: u64, step: u64) -> StableTorchResult<Tensor> {
        // This is all on the cpu... but we can also use a kernel;
        // https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L795

        let options: EmtpyOptions = EmtpyOptions {
            dtype: Some(ScalarType::UInt64),
            ..Default::default()
        };
        let mut v = Tensor::zeros(&[((end - start) / step) as usize], &options)?;
        let mut value = start;
        for i in 0..v.sizes()[0] {
            v.t_mut::<u64>()?[i] = value;
            value += step;
        }
        Ok(v)
    }
}

pub trait TensorPropertiesContrib {
    fn storage_size(&self) -> usize;
}
impl TensorPropertiesContrib for Tensor {
    fn storage_size(&self) -> usize {
        let mut storage_size: i64 = 0;
        unsafe_call_panic!(aoti_torch_get_storage_size(self.get(), &mut storage_size));
        if storage_size < 0 {
            panic!("storage_offset got negative value");
        }
        storage_size as usize
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::stable::ops::{EmtpyOptions, ToOptions};

    use crate::headeronly::core::ScalarType;

    use crate::stable::device::{Device, DeviceIndex};
    #[test]
    fn test_tensor_conbtrib_from_to_scalar() {
        use crate::contrib::TensorFromScalar;
        use crate::contrib::TensorToScalar;
        let t = Tensor::from_f32(std::f32::consts::PI).unwrap();
        assert_eq!(t.dim(), 0);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.sizes(), &[]);
        assert_eq!(t.strides(), &[]);
        assert_eq!(t.is_contiguous(), true);
        // assert_eq!(t.stride(0), 0); // no dimensions
        assert_eq!(t.get_device_index(), DeviceIndex(-1));
        assert_eq!(t.defined(), true);
        assert_eq!(t.scalar_type(), ScalarType::Float);
        assert_eq!(t.device(), Device::try_from("cpu").unwrap());
        assert_eq!(t.to_f32().unwrap(), std::f32::consts::PI);

        // and double

        let t = Tensor::from_f64(std::f64::consts::PI).unwrap();
        assert_eq!(t.scalar_type(), ScalarType::Double);
        assert_eq!(t.to_f64().unwrap(), std::f64::consts::PI);
    }

    #[test]
    fn test_tensor_contrib_addition() -> StableTorchResult<()> {
        use crate::contrib::{TensorFromScalar, TensorToScalar};
        let a = Tensor::from_f32(5.0)?;
        let b = Tensor::from_f32(3.0)?;
        let c = a.add(&b)?;
        assert_eq!(c.to_f32().unwrap(), 8.0);
        Ok(())
    }

    #[test]
    fn test_tensor_contrib_data() -> StableTorchResult<()> {
        let mut a = Tensor::from_f32(5.0)?.unsqueeze(0)?;
        println!("a.element_size(): {:?}", a.element_size());
        println!("a.scalar_type(): {:?}", a.scalar_type());
        println!("a.const_data_ptr: {:?}", a.const_data_ptr());
        println!("a.mutable_data_ptr: {:?}", a.mutable_data_ptr());
        println!("a.data_mut(): {:?}", a.data_mut()?);
        a.data_mut()?[0] = 3;
        println!("a.data_ref(): {:?}", a.data_ref()?);
        assert_eq!(a.data_ref()?, &[3, 0, 160, 64]);

        assert_eq!(a.data_ref()?, a.t_ref::<u8>()?);

        Ok(())
    }

    #[test]
    fn test_tensor_contrib_creation() -> StableTorchResult<()> {
        let t = Tensor::zeros(
            &[2, 2],
            &EmtpyOptions {
                dtype: Some(ScalarType::Float),
                ..Default::default()
            },
        )?;
        println!("t.data_ptr: {:?}", t.const_data_ptr());
        println!("t.f32_ref(): {:?}", t.f32_ref()?);
        let s = t.f32_ref()?;
        assert_eq!(s, &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(t.dim(), 2);
        assert_eq!(t.sizes(), &[2, 2]);

        Ok(())
    }
    #[test]
    fn test_tensor_contrib_f32() -> StableTorchResult<()> {
        let mut t = Tensor::zeros(
            &[2, 2],
            &EmtpyOptions {
                dtype: Some(ScalarType::Float),
                ..Default::default()
            },
        )?;
        println!("t.const_data_ptr: {:?}", t.const_data_ptr());
        println!("t.mutable_data_ptr: {:?}", t.mutable_data_ptr());
        // println!("t.f32_ref(): {:?}", t.f32_ref()?); yes this is unitialised, lets not look at it as valgrind lights up.
        t.f32_mut()?[0] = 3.3;
        t.f32_mut()?[1] = 5.3;
        t.f32_mut()?[2] = 89.5;
        t.f32_mut()?[3] = 1.5;
        println!("t.f32_ref(): {:?}", t.f32_ref()?);

        // This definitely shares storage;
        let b = t.to(&ToOptions {
            device: Some(Device::try_from("cpu")?),
            //device: Some(Device::from_str("cuda:0")?),
            copy: false,
            ..Default::default()
        })?;
        println!("before: b.f32_ref(): {:?}", b.f32_ref()?);
        t.f32_mut()?[3] = 10000.5;
        println!("after   b.f32_ref(): {:?}", b.f32_ref()?);

        Ok(())
    }
    #[test]
    fn test_tensor_contrib_arange() -> StableTorchResult<()> {
        let t = Tensor::arange_u64(0, 24, 1)?;
        assert_eq!(t.dim(), 1); // torch.arange(0, 24).dim()
        assert_eq!(t.sizes(), &[24]); // torch.arange(0, 24).shape
        assert!(t.t_ref::<u64>()?.iter().zip(0..24u64).all(|(a, b)| *a == b));

        Ok(())
    }

    #[cfg(feature = "cuda")]
    // #[test]
    #[allow(dead_code)]
    fn test_tensor_contrib_cuda_stream_tests() -> StableTorchResult<()> {
        use crate::aoti_torch::*;
        use crate::stable::c::*;
        let a = Tensor::zeros(
            &[20000, 20000],
            &EmtpyOptions {
                device: Some(Device::CUDA),
                ..Default::default()
            },
        )?;
        let b = a.matmul(&a)?;

        let mut ret_device_index: i32 = 0;

        unsafe_call_bail!(aoti_torch_get_current_device_index(&mut ret_device_index,));
        println!("ret_device_index: {ret_device_index:?}");

        let device_index = 0;

        let mut handle_stream: StreamHandle = std::ptr::null_mut();

        unsafe_call_bail!(aoti_torch_get_current_stream(
            device_index,
            &mut handle_stream
        ));
        assert!(!handle_stream.is_null());
        println!("handle_stream: {handle_stream:?}");

        // get a native handle;

        let mut ret_stream: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_get_current_cuda_stream(
            device_index,
            &mut ret_stream
        ));
        assert!(!ret_stream.is_null());
        println!("ret_stream: {ret_stream:?}");

        let mut native_stream: *mut std::ffi::c_void = std::ptr::null_mut();

        unsafe_call_bail!(torch_stream_native_handle(
            handle_stream,
            &mut native_stream
        ));
        assert!(!native_stream.is_null());
        println!("native_stream: {native_stream:?}");

        // unsafe_call_bail!(torch_cuda_stream_synchronize(
        //     handle_stream.cast(),
        //     device_index
        // ));
        println!("b: {:?}", b.sizes());

        Ok(())
    }
}
