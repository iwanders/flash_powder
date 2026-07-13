//! This holds functions that pytorch puts into the functional module.
use crate::{properties::TensorProperties, DType};
use anyhow::bail;
use torch_stable::{
    aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor, unsafe_call_dispatch_bail,
};

// Should all these (i64, i64) be arrays instead?
// What about the variable length arrays in interpolate? Currently they are [f32;3]... but in python you can provide a
// scalar or even just two values instead of all three.

use crate::{StableTorchResult, Tensor, TensorAccess};
/// Options for conv2d.
#[derive(Copy, Clone, Debug)]
pub struct Conv2dOptions {
    pub stride: (i64, i64),
    pub padding: (i64, i64),
    pub dilation: (i64, i64),
    pub groups: i64,
}
impl Default for Conv2dOptions {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

/// A 2d convolution.
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1757)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.conv2d.html)
pub fn conv2d<I: TensorAccess, W: TensorAccess>(
    input: &I,
    weight: &W,
    bias: Option<&W>,
    options: &Conv2dOptions,
) -> StableTorchResult<Tensor> {
    // return wrap(dispatch_conv2d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
    let mut stack: [StableIValue; 7] = [
        input.get_tensor().into(),
        weight.get_tensor().into(),
        (&bias.map(|z| z.get_tensor())).into(),
        (&options.stride).into(),
        (&options.padding).into(),
        (&options.dilation).into(),
        (&options.groups).into(),
    ];
    unsafe_call_dispatch_bail!("aten::conv2d", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;

    Ok(Tensor::new(r))
}

/// Options for conv2d transpose.
#[derive(Copy, Clone, Debug)]
pub struct ConvTranspose2dOptions {
    pub stride: (i64, i64),
    pub padding: (i64, i64),
    pub output_padding: (i64, i64),
    pub groups: i64,
    pub dilation: (i64, i64),
}
impl Default for ConvTranspose2dOptions {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

///  conv_transpose2d
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L1793)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.conv_transpose2d.html)
pub fn conv_transpose2d<T: TensorAccess>(
    input: &T,
    weight: &T,
    bias: Option<&T>,
    options: &ConvTranspose2dOptions,
) -> StableTorchResult<Tensor> {
    // Comment on https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L1788
    // is nice :)
    let mut stack: [StableIValue; 8] = [
        input.get_tensor().into(),
        weight.get_tensor().into(),
        (&bias.map(|z| z.get_tensor())).into(),
        (&options.stride).into(),
        (&options.padding).into(),
        (&options.output_padding).into(),
        (&options.groups).into(),
        (&options.dilation).into(),
    ];
    unsafe_call_dispatch_bail!("aten::conv_transpose2d", "input", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;

    Ok(Tensor::new(r))
}

/// Options for max_pool2d.
#[derive(Copy, Clone, Debug)]
pub struct MaxPool2dDOptions {
    /// Stride to use, if unset defaults to kernel_size.
    pub stride: Option<(i64, i64)>,
    pub padding: (i64, i64),
    pub dilation: (i64, i64),
    pub ceil_mode: bool,
}
impl Default for MaxPool2dDOptions {
    fn default() -> Self {
        Self {
            stride: None,
            padding: (0, 0),
            dilation: (1, 1),
            ceil_mode: false,
        }
    }
}
/// Maxpool 2d
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L3991)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.max_pool2d.html)
pub fn max_pool2d<T: TensorAccess>(
    input: &T,
    kernel_size: (i64, i64),
    options: &MaxPool2dDOptions,
) -> StableTorchResult<Tensor> {
    let stride = options.stride.unwrap_or(kernel_size);
    let mut stack: [StableIValue; 6] = [
        input.get_tensor().into(),
        (&kernel_size).into(),
        (&stride).into(),
        (&options.padding).into(),
        (&options.dilation).into(),
        (options.ceil_mode).into(),
    ];
    unsafe_call_dispatch_bail!("aten::max_pool2d", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;

    Ok(Tensor::new(r))
}

/// Relu
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L5251)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.relu.html)
pub fn relu<T: TensorAccess>(input: &T) -> StableTorchResult<Tensor> {
    let mut stack: [StableIValue; 1] = [input.get_tensor().into()];
    unsafe_call_dispatch_bail!("aten::relu", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;
    assert_ne!(input.get_tensor().const_data_ptr(), r.const_data_ptr());

    Ok(Tensor::new(r))
}

/// Linear
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L3419)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.linear.html)
pub fn linear<I: TensorAccess, W: TensorAccess, B: TensorAccess>(
    input: &I,
    weight: &W,
    bias: Option<&B>,
) -> StableTorchResult<Tensor> {
    let mut stack: [StableIValue; 3] = [
        input.get_tensor().into(),
        weight.get_tensor().into(),
        (&bias.map(|z| z.get_tensor())).into(),
    ];
    unsafe_call_dispatch_bail!("aten::linear", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;
    Ok(Tensor::new(r))
}

/// Adaptive Avg Pool2d
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L12497)
/// - [pytorch functional](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_avg_pool2d.html)
/// - [pytorch nn](https://docs.pytorch.org/docs/2.11/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d)
pub fn adaptive_avg_pool2d<I: TensorAccess>(
    input: &I,
    output: (i64, i64),
) -> StableTorchResult<Tensor> {
    let mut stack: [StableIValue; 2] = [input.get_tensor().into(), (&output).into()];
    unsafe_call_dispatch_bail!("aten::adaptive_avg_pool2d", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;
    Ok(Tensor::new(r))
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum InterpolateAlgorithm {
    #[default]
    Nearest,
    Linear,
    Bilinear,
    Trilinear,
    Bicubic,
    Area,
    NearestExact,
}

/// Options for interpolate.
///
/// If lower dimensionality, only the first values are used.
#[derive(Copy, Clone, Debug)]
pub struct InterpolateOptions {
    pub size: Option<[i64; 3]>,
    pub scale_factor: Option<[f64; 3]>,
    pub mode: InterpolateAlgorithm,
    pub align_corners: Option<bool>,
    pub recompute_scale_factor: Option<bool>,
    pub antialias: bool,
}
impl Default for InterpolateOptions {
    fn default() -> Self {
        Self {
            size: None,
            scale_factor: None,
            mode: InterpolateAlgorithm::Nearest,
            recompute_scale_factor: None,
            align_corners: None,
            antialias: false,
        }
    }
}
pub fn interpolate<T: TensorAccess + TensorProperties>(
    input: &T,
    options: &InterpolateOptions,
) -> StableTorchResult<Tensor> {
    let dim = input.dim() - 2; // Number of spatial dimensions.
                               // Validation in https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/functional.py#L4715-L4761 :o

    let mut scale_factors: Option<&[f64]> = None;
    let mut output_size: Option<&[i64]> = None;
    let mode = options.mode;

    let mut align_corners: bool = false;
    if mode == InterpolateAlgorithm::Nearest
        || mode == InterpolateAlgorithm::Area
        || mode == InterpolateAlgorithm::NearestExact
    {
        if options.align_corners.is_some() {
            bail!(
                "align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear"
            )
        }
    } else {
        align_corners = options.align_corners.unwrap_or_default();
    }

    if options.size.is_some() && options.scale_factor.is_some() {
        bail!("only one of size or scale_factor should be defined");
    } else if options.size.is_some() {
        // We can't validate lengths here because our arrays are always fixed size.
        output_size = Some(&options.size.as_ref().unwrap()[0..dim]);
    } else if let Some(scale_factor) = options.scale_factor.as_ref() {
        // We can't validate lengths here because our arrays are always fixed size.
        scale_factors = Some(&scale_factor[0..dim]);
    } else {
        bail!("one of size or scale_factor must be defined");
    }

    if options.recompute_scale_factor.is_some() {
        todo!("recomputing scale factor not yet supported");
    }
    if options.antialias
        && !((mode == InterpolateAlgorithm::Bilinear || mode == InterpolateAlgorithm::Bicubic)
            && input.dim() == 4)
    {
        bail!(
            "Anti-alias option is restricted to bilinear and bicubic modes and requires a 4-D tensor as input"
        );
    }

    macro_rules! dispatch_interpolate_simple {
        ($t:literal ) => {{
            let mut stack: [StableIValue; 3] = [
                input.get_tensor().into(),
                (&output_size).into(),
                (&scale_factors).into(),
            ];
            unsafe_call_dispatch_bail!($t, "vec", stack.as_mut_slice());
            let r: StableTensor = stack[0].try_into()?;

            return Ok(Tensor::new(r));
        }};
    }

    macro_rules! dispatch_interpolate_corners {
        ($t:literal) => {{
            let mut stack: [StableIValue; 4] = [
                input.get_tensor().into(),
                (&output_size).into(),
                (align_corners).into(),
                (&scale_factors).into(),
            ];
            unsafe_call_dispatch_bail!($t, "vec", stack.as_mut_slice());
            let r: StableTensor = stack[0].try_into()?;

            return Ok(Tensor::new(r));
        }};
    }

    // This indeed is a forest of if statements; https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/functional.py#L4812-L4830
    if input.dim() == 3 && options.mode == InterpolateAlgorithm::Nearest {
        dispatch_interpolate_simple!("aten::upsample_nearest1d");
    } else if input.dim() == 4 && options.mode == InterpolateAlgorithm::Nearest {
        dispatch_interpolate_simple!("aten::upsample_nearest2d");
    } else if input.dim() == 5 && options.mode == InterpolateAlgorithm::Nearest {
        dispatch_interpolate_simple!("upsample_nearest3d");
    }

    if input.dim() == 3 && options.mode == InterpolateAlgorithm::Linear {
        dispatch_interpolate_corners!("aten::upsample_linear1d");
    } else if input.dim() == 4 && options.mode == InterpolateAlgorithm::Bilinear {
        dispatch_interpolate_corners!("aten::upsample_bilinear2d");
    } else if input.dim() == 5 && options.mode == InterpolateAlgorithm::Trilinear {
        dispatch_interpolate_corners!("aten::upsample_trilinear3d");
    }

    todo!("missing some of the complex branches in interpolate")
}

/// Options for upsample.
///
/// If lower dimensionality, only the first values are used.
#[derive(Copy, Clone, Debug)]
pub struct UpsampleOptions {
    pub size: Option<[i64; 3]>,
    pub scale_factor: Option<[f64; 3]>,
    pub mode: InterpolateAlgorithm,
    pub align_corners: Option<bool>,
}
impl Default for UpsampleOptions {
    fn default() -> Self {
        Self {
            size: None,
            scale_factor: None,
            mode: InterpolateAlgorithm::Nearest,
            align_corners: None,
        }
    }
}
/// Upsample
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L13001-L13053)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.upsample.html)
/// - [pytorch class](https://docs.pytorch.org/docs/2.11/generated/torch.nn.modules.upsampling.Upsample.html)
pub fn upsample<T: TensorAccess + TensorProperties>(
    input: &T,
    options: &UpsampleOptions,
) -> StableTorchResult<Tensor> {
    // Interestingly; https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/modules/upsampling.py#L174
    // this calls interpolate under the hood...
    let interpolate_options = InterpolateOptions {
        size: options.size,
        scale_factor: options.scale_factor,
        mode: options.mode,
        recompute_scale_factor: None,
        align_corners: options.align_corners,
        antialias: false,
    };

    interpolate(input, &interpolate_options)
}

/// Softmax
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L13001-L13053)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.softmax.html)
/// - [pytorch class](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Softmax.html#torch.nn.Softmax)
///
/// softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
pub fn softmax_int<T: TensorAccess + TensorProperties>(
    input: &T,
    dim: isize,
    dtype: Option<DType>,
) -> StableTorchResult<Tensor> {
    let mut stack: [StableIValue; 3] = [input.get_tensor().into(), (dim).into(), (&dtype).into()];
    unsafe_call_dispatch_bail!("aten::softmax", "int", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;

    Ok(Tensor::new(r))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_conv2d() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([1,4,4])
            w = torch.tensor([[[1.0, 2.0],[3.0, 4.0]]]).unsqueeze(0)
            r = torch.nn.functional.conv2d(d, w)
        */

        let d = Tensor::from(&[[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]])?;
        assert_eq!(d.sizes(), &[1, 4, 4]); // #PYTHON list(d.shape)
        assert_eq!(
            d.f32s_ref()?,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())

        let mut w = Tensor::zeros(&[1, 1, 2, 2], &Default::default())?;
        for (i, v) in w.ds_mut::<f32>()?.iter_mut().enumerate() {
            *v = (i + 1) as f32
        }
        assert_eq!(w.sizes(), &[1, 1, 2, 2]);
        assert_eq!(w.f32s_ref()?, &[1.0f32, 2.0, 3.0, 4.0]); // #PYTHON list(w.view(-1).tolist())

        let r = conv2d(&d, &w, None, &Default::default())?;
        assert_eq!(r.sizes(), &[1, 3, 3]); // #PYTHON list(r.shape)

        assert_eq!(
            r.f32s_ref()?,
            &[44.0f32, 54.0, 64.0, 84.0, 94.0, 104.0, 124.0, 134.0, 144.0]
        ); // #PYTHON list(r.view(-1).tolist())

        /*
            #|PYTHON
            r = torch.nn.functional.conv2d(d, w, stride=(1, 2))
        */
        let r = conv2d(
            &d,
            &w,
            None,
            &Conv2dOptions {
                stride: (1, 2),
                ..Default::default()
            },
        )?;
        assert_eq!(r.sizes(), &[1, 3, 2]); // #PYTHON list(r.shape)
        assert_eq!(r.f32s_ref()?, &[44.0f32, 64.0, 84.0, 104.0, 124.0, 144.0]); // #PYTHON list(r.view(-1).tolist())

        /*
            #|PYTHON
            r = torch.nn.functional.conv2d(d, w, stride=(2, 2))
        */
        let r = conv2d(
            &d,
            &w,
            None,
            &Conv2dOptions {
                stride: (2, 2),
                ..Default::default()
            },
        )?;
        assert_eq!(r.sizes(), &[1, 2, 2]); // #PYTHON list(r.shape)
        assert_eq!(r.f32s_ref()?, &[44.0f32, 64.0, 124.0, 144.0]); // #PYTHON list(r.view(-1).tolist())

        /*
            #|PYTHON
            r = torch.nn.functional.conv2d(d, w, padding=(1, 2))
        */
        let r = conv2d(
            &d,
            &w,
            None,
            &Conv2dOptions {
                padding: (1, 2),
                ..Default::default()
            },
        )?;
        assert_eq!(r.sizes(), &[1, 5, 7]); // #PYTHON list(r.shape)
        assert_eq!(
            r.f32s_ref()?,
            &[
                0.0f32, 4.0, 11.0, 18.0, 25.0, 12.0, 0.0, 0.0, 22.0, 44.0, 54.0, 64.0, 28.0, 0.0,
                0.0, 46.0, 84.0, 94.0, 104.0, 44.0, 0.0, 0.0, 70.0, 124.0, 134.0, 144.0, 60.0, 0.0,
                0.0, 26.0, 41.0, 44.0, 47.0, 16.0, 0.0
            ]
        ); // #PYTHON list(r.view(-1).tolist())

        /*
            #|PYTHON
            b = torch.tensor([5.0])
            r = torch.nn.functional.conv2d(d, w, b)
        */
        let mut b = Tensor::zeros(&[1], &Default::default())?;
        b.f32s_mut()?.copy_from_slice(&[5.0]);
        assert_eq!(b.sizes(), &[1]); // #PYTHON list(b.shape)

        let r = conv2d(&d, &w, Some(&b), &Default::default())?;
        assert_eq!(r.sizes(), &[1, 3, 3]); // #PYTHON list(r.shape)
        assert_eq!(
            r.f32s_ref()?,
            &[49.0f32, 59.0, 69.0, 89.0, 99.0, 109.0, 129.0, 139.0, 149.0]
        ); // #PYTHON list(r.view(-1).tolist())

        Ok(())
    }

    #[test]
    fn test_flash_powder_conv_transpose2d() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([1,4,4])
            w = torch.tensor([[[1.0, 2.0],[3.0, 4.0]]]).unsqueeze(0)
            r = torch.nn.functional.conv_transpose2d(d, w)
        */

        let d = Tensor::from(&[[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]])?;
        assert_eq!(d.sizes(), &[1, 4, 4]); // #PYTHON list(d.shape)
        assert_eq!(
            d.f32s_ref()?,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())

        let mut w = Tensor::zeros(&[1, 1, 2, 2], &Default::default())?;
        for (i, v) in w.ds_mut::<f32>()?.iter_mut().enumerate() {
            *v = (i + 1) as f32
        }
        assert_eq!(w.sizes(), &[1, 1, 2, 2]);
        assert_eq!(w.f32s_ref()?, &[1.0f32, 2.0, 3.0, 4.0]); // #PYTHON list(w.view(-1).tolist())

        let r = conv_transpose2d(&d, &w, None, &Default::default())?;
        assert_eq!(r.sizes(), &[1, 5, 5]); // #PYTHON list(r.shape)

        assert_eq!(
            r.f32s_ref()?,
            &[
                1.0f32, 4.0, 7.0, 10.0, 8.0, 8.0, 26.0, 36.0, 46.0, 32.0, 24.0, 66.0, 76.0, 86.0,
                56.0, 40.0, 106.0, 116.0, 126.0, 80.0, 39.0, 94.0, 101.0, 108.0, 64.0
            ]
        ); // #PYTHON list(r.view(-1).tolist())
        Ok(())
    }

    #[test]
    fn test_flash_powder_relu() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=torch.float)
            r = d.relu()
        */
        let d = Tensor::from(&[-1.0f32, 0.0, 0.5, 1.0])?;

        assert_eq!(d.sizes(), &[4]); // #PYTHON list(d.shape)
        let r = relu(&d)?;
        assert_eq!(r.f32s_ref()?, &[0.0f32, 0.0, 0.5, 1.0]); // #PYTHON list(r.view(-1).tolist())

        Ok(())
    }
    #[test]
    fn test_flash_power_max_pool2d() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([1,4,4])
            r = torch.nn.functional.max_pool2d(d, (2,2))
        */
        let d = Tensor::from(&[[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]])?;

        assert_eq!(d.sizes(), &[1, 4, 4]); // #PYTHON list(d.shape)
        assert_eq!(
            d.f32s_ref()?,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())

        let r = max_pool2d(&d, (2, 2), &Default::default())?;
        assert_eq!(r.sizes(), &[1, 2, 2]); // #PYTHON list(r.shape)
        assert_eq!(r.f32s_ref()?, &[6.0f32, 8.0, 14.0, 16.0]); // #PYTHON list(r.view(-1).tolist())

        /*
            #|PYTHON
            r = torch.nn.functional.max_pool2d(d, (2,2), stride=3)
        */
        let r = max_pool2d(
            &d,
            (2, 2),
            &MaxPool2dDOptions {
                stride: Some((3, 3)),
                ..Default::default()
            },
        )?;
        assert_eq!(r.sizes(), &[1, 1, 1]); // #PYTHON list(r.shape)
        assert_eq!(r.f32s_ref()?, &[6.0f32]); // #PYTHON list(r.view(-1).tolist())

        Ok(())
    }

    #[test]
    fn test_flash_powder_upsample() -> StableTorchResult<()> {
        // Example values from https://docs.pytorch.org/docs/2.11/generated/torch.nn.modules.upsampling.Upsample.html
        /*
            #|PYTHON
            input = torch.tensor(list(range(1,5)), dtype=torch.float).reshape([1,1, 2,2])
        */
        let d = Tensor::from(&[[1.0f32, 2.0], [3.0, 4.0]])?
            .view(&[1, 1, 2, 2])?
            .to_owned()?;
        assert_eq!(d.sizes(), &[1, 1, 2, 2]); // #PYTHON list(input.shape)
        assert_eq!(d.f32s_ref()?, &[1.0f32, 2.0, 3.0, 4.0]); // #PYTHON list(input.view(-1).tolist())

        // Nearest 2
        /*
            #|PYTHON
            m = torch.nn.functional.interpolate(input, scale_factor=2, mode="nearest")
        */
        let m = upsample(
            &d,
            &UpsampleOptions {
                scale_factor: Some([2.0, 2.0, 2.0]),
                mode: InterpolateAlgorithm::Nearest,
                ..Default::default()
            },
        )?;
        assert_eq!(m.sizes(), &[1, 1, 4, 4]); // #PYTHON list(m.shape)
        assert_eq!(
            m.f32s_ref()?,
            &[1.0f32, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]
        ); // #PYTHON list(m.view(-1).tolist())

        // Bilinear 2
        /*
            #|PYTHON
            m = torch.nn.functional.interpolate(input, scale_factor=2, mode="bilinear")
        */
        let m = upsample(
            &d,
            &UpsampleOptions {
                scale_factor: Some([2.0, 2.0, 2.0]),
                mode: InterpolateAlgorithm::Bilinear,
                ..Default::default()
            },
        )?;
        assert_eq!(m.sizes(), &[1, 1, 4, 4]); // #PYTHON list(m.shape)
        assert_eq!(
            m.f32s_ref()?,
            &[
                1.0f32, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25,
                3.75, 4.0
            ]
        ); // #PYTHON list(m.view(-1).tolist())

        // Bilinear 2, aligned corners.
        /*
            #|PYTHON
            m = torch.nn.functional.interpolate(input, scale_factor=2, mode="bilinear", align_corners=True)
        */
        let m = upsample(
            &d,
            &UpsampleOptions {
                scale_factor: Some([2.0, 2.0, 2.0]),
                mode: InterpolateAlgorithm::Bilinear,
                align_corners: Some(true),
                ..Default::default()
            },
        )?;
        assert_eq!(m.sizes(), &[1, 1, 4, 4]); // #PYTHON list(m.shape)
        assert_eq!(
            m.f32s_ref()?,
            &[
                1.0f32,
                1.3333332538604736,
                1.6666667461395264,
                2.0,
                1.6666666269302368,
                2.0,
                2.3333334922790527,
                2.6666665077209473,
                2.3333334922790527,
                2.6666665077209473,
                3.0,
                3.3333334922790527,
                3.0,
                3.3333332538604736,
                3.6666667461395264,
                4.0
            ]
        ); // #PYTHON list(m.view(-1).tolist())

        // For the rest of the examples from Upsample we need slicing to do that copy... which we don't currently have.
        // TODO

        Ok(())
    }

    #[test]
    fn test_flash_powder_linear() -> StableTorchResult<()> {
        /*
            #|PYTHON
            w = torch.tensor(list(range(1,10)), dtype=torch.float).reshape([3,3])
            b = torch.tensor([1.0,2.0,3.0])
            x = torch.tensor([5.0,6.0,7.0])
            v = torch.nn.functional.linear(x, w, b)
        */

        let w = Tensor::from(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])?;
        let b = Tensor::from(&[1.0f32, 2.0, 3.0])?;
        let x: Tensor = [5.0f32, 6.0, 7.0].try_into()?;
        assert_eq!(w.sizes(), &[3, 3]); // #PYTHON list(w.shape)
        assert_eq!(
            w.f32s_ref()?,
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ); // #PYTHON list(w.view(-1).tolist())

        assert_eq!(b.sizes(), &[3]); // #PYTHON list(b.shape)
        assert_eq!(b.f32s_ref()?, &[1.0f32, 2.0, 3.0]); // #PYTHON list(b.view(-1).tolist())
        assert_eq!(x.sizes(), &[3]); // #PYTHON list(x.shape)
        assert_eq!(x.f32s_ref()?, &[5.0f32, 6.0, 7.0]); // #PYTHON list(x.view(-1).tolist())

        let v = linear(&x, &w, Some(&b))?;
        assert_eq!(v.sizes(), &[3]); // #PYTHON list(v.shape)
        assert_eq!(v.f32s_ref()?, &[39.0f32, 94.0, 149.0]); // #PYTHON list(v.view(-1).tolist())

        Ok(())
    }

    #[test]
    fn test_flash_powder_adaptive_avg_pool2d() -> StableTorchResult<()> {
        // Example from https://docs.pytorch.org/docs/2.11/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d
        /*
            #|PYTHON
            w = torch.randn(1, 64, 8, 9)
            m = torch.nn.AdaptiveAvgPool2d((5,7))
            output = m(w)
        */
        let w = Tensor::randn(&[1, 64, 8, 9], &Default::default())?;

        assert_eq!(w.sizes(), &[1, 64, 8, 9]); // #PYTHON list(w.shape)
        let output = adaptive_avg_pool2d(&w, (5, 7))?;
        assert_eq!(output.sizes(), &[1, 64, 5, 7]); // #PYTHON list(output.shape)
        Ok(())
    }

    #[test]
    fn test_flash_powder_softmax_int() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,10)), dtype=torch.float).reshape([1,3,3])
            s0 = torch.nn.functional.softmax(d, 0)
            s1 = torch.nn.functional.softmax(d, 1)
        */

        let d = Tensor::from(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])?;
        assert_eq!(d.sizes(), &[1, 3, 3]); // #PYTHON list(d.shape)

        let s0 = softmax_int(&d, 0, None)?;
        assert_eq!(s0.sizes(), &[1, 3, 3]); // #PYTHON list(s0.shape)
        assert_eq!(
            s0.f32s_ref()?,
            &[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ); // #PYTHON list(s0.view(-1).tolist())
        let s1 = softmax_int(&d, 1, None)?;
        assert_eq!(s1.sizes(), &[1, 3, 3]); // #PYTHON list(s1.shape)
        assert_eq!(
            s1.f32s_ref()?,
            &[
                0.0023556333035230637f32,
                0.0023556333035230637,
                0.0023556333035230637,
                0.04731415584683418,
                0.04731415584683418,
                0.04731415584683418,
                0.9503302574157715,
                0.9503302574157715,
                0.9503302574157715
            ]
        ); // #PYTHON list(s1.view(-1).tolist())

        Ok(())
    }
}
