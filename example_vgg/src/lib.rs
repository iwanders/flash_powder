/// Quickly thrown together example of VGG11
///
/// `https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html#torchvision.models.vgg11`
/// Code:
/// `https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L91`.
///
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py

// Smallest is vgg11;
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L306-L329
// which is of the 'A' category.
//    weights = VGG11_Weights.verify(weights)
// return _vgg("A", False, weights, progress, **kwargs)
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L98
// VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
// batch norm = False
// Layers per 'A'.
// We're doing inference, we can skip the dropout.
use anyhow::bail;
use safetensors::SafeTensors;

use flash_powder as fp;
use flash_powder::functional;
use flash_powder::nn;
use flash_powder::prelude::*;
use fp::{DType, Device, Ten, Tensor};

// Bit of tooling for type erasure.
use nn::Module as ForwardLayer;
type ForwardFun = dyn Fn(&Ten) -> Result<Tensor, anyhow::Error>;

struct LambdaForward {
    f: Box<ForwardFun>,
}
impl std::fmt::Debug for LambdaForward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LambdaForward").finish()
    }
}

impl LambdaForward {
    fn new<F>(f: F) -> Self
    where
        F: Fn(&Ten) -> Result<Tensor, anyhow::Error> + 'static,
    {
        Self { f: Box::new(f) }
    }
}
impl ForwardLayer for LambdaForward {
    fn forward(&self, tensor: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        (self.f)(tensor)
    }
}

// The config, as per https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L91
// but then as integers.
const CFG_A: &[u32] = &[
    64, 'M' as u32, 128, 'M' as u32, 256, 256, 'M' as u32, 512, 512, 'M' as u32, 512, 512,
    'M' as u32,
];

/// VGG struct with layers and classifier.
pub struct VGG {
    features: nn::Sequential,
    classifier: nn::Sequential,
}
impl VGG {
    /// Create a new vgg config as per the layer specification and the provided weights.
    pub fn new(cfg: &[u32], tensors: &SafeTensors, use_cuda: bool) -> Result<Self, anyhow::Error> {
        const PRINT_TENSOR_SHAPES: bool = false;

        if PRINT_TENSOR_SHAPES {
            let mut sorted = tensors.names().clone();
            sorted.sort();

            for name in sorted {
                println!("Tensor key: {}", name);

                if let Ok(view) = tensors.tensor(name) {
                    println!("  Shape: {:?}", view.shape());
                }
            }
        }

        // Okay... so we iterate over cfg... build the layers and then append them to ourselves? How hard can it be.
        let mut layers: Vec<Box<dyn ForwardLayer>> = vec![];

        // First, the group called 'features'.
        let mut feature_counter = 0;
        let mut in_channels = 3;
        for v in cfg.iter() {
            let layer_name = format!("features.{feature_counter}");
            if *v == 'M' as u32 {
                // should be maxpool2d.
                layers.push(Box::new(nn::MaxPool2d {
                    kernel_size: (2, 2),
                    options: Default::default(),
                }));
                feature_counter += 1;
            } else {
                let conv2d_options = functional::Conv2dOptions {
                    stride: (1, 1),
                    padding: (1, 1),
                    ..Default::default()
                };
                let layer = nn::Conv2d::new(in_channels, *v as usize, (3, 3), conv2d_options)?;

                in_channels = *v as usize;
                layers.push(create_conv2d(tensors, &layer_name, use_cuda)?);
                // + 1 for conv2d, +1 for relu.
                feature_counter += 2;
            }
        }
        let features = layers.drain(..).collect();

        // https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L43
        // and then the group called classifier
        let mut classifier: nn::Sequential = Default::default();
        classifier.push(create_linear(tensors, &format!("classifier.0"), use_cuda)?);
        classifier.push(nn::ReLU.into_boxed());
        // dropout

        // next linear block
        classifier.push(create_linear(tensors, &format!("classifier.3"), use_cuda)?);
        classifier.push(nn::ReLU.into_boxed());
        // dropout

        // last linear.
        classifier.push(create_linear(tensors, &format!("classifier.6"), use_cuda)?);

        Ok(VGG {
            features,
            classifier,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, anyhow::Error> {
        let mut r: Tensor = input.clone();
        // Run it through the layers.
        r = self.features.forward(&r.ten()?)?;

        // do some avgpool.
        r = fp::functional::adaptive_avg_pool2d(&r, (7, 7))?;
        // do a flatten.
        r = r.flatten(1, None)?;

        // Run it through the classifier.
        r = self.classifier.forward(&r.ten()?)?;

        Ok(r)
    }

    pub fn to_cuda(&mut self) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

fn optional_cuda_to(v: Tensor, use_cuda: bool) -> Result<Tensor, anyhow::Error> {
    if use_cuda {
        v.to(&Device::CUDA.into())
    } else {
        Ok(v)
    }
}

/// Helper to read a linear layer from the safetensors.
fn create_linear(
    tensors: &SafeTensors,
    layer_name: &str,
    use_cuda: bool,
) -> Result<Box<dyn ForwardLayer>, anyhow::Error> {
    let weights = optional_cuda_to(
        safetensor_to_tensor(tensors, &format!("{layer_name}.weight"))?,
        use_cuda,
    )?;
    let bias = {
        if let Ok(v) = safetensor_to_tensor(tensors, &format!("{layer_name}.bias")) {
            Some(optional_cuda_to(v, use_cuda)?)
        } else {
            None
        }
    };
    Ok(Box::new(LambdaForward::new(move |input: &Ten| {
        let bias_ref = bias.as_ref();
        let res = functional::linear(input, &weights, bias_ref)?;
        Ok(res)
    })))
}

/// Helper to read a conv2d layer from the safetensors.
fn create_conv2d(
    tensors: &SafeTensors,
    layer_name: &str,
    use_cuda: bool,
) -> Result<Box<dyn ForwardLayer>, anyhow::Error> {
    let conv2d_options = functional::Conv2dOptions {
        stride: (1, 1),
        padding: (1, 1),
        ..Default::default()
    };
    //
    let weights = optional_cuda_to(
        safetensor_to_tensor(tensors, &format!("{layer_name}.weight"))?,
        use_cuda,
    )?;
    let bias = {
        if let Ok(v) = safetensor_to_tensor(tensors, &format!("{layer_name}.bias")) {
            Some(optional_cuda_to(v, use_cuda)?)
        } else {
            None
        }
    };
    Ok(Box::new(LambdaForward::new(move |t: &Ten| {
        let bias_ref = bias.as_ref();
        let conv_res = functional::conv2d(t, &weights, bias_ref, &conv2d_options)?;
        let res = functional::relu(&conv_res)?;

        Ok(res)
    })))
}

/// Converter to go from safetnsors DType to flash_powder DType
fn safetensor_dtype_to_scalar_type(v: safetensors::Dtype) -> DType {
    match v {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F64 => DType::F64,
        _ => todo!("todo handle {v:?}"),
    }
}

/// Convert a tensor by `name` from `tensors` into a flash powder Tensor.
fn safetensor_to_tensor(tensors: &SafeTensors, name: &str) -> Result<Tensor, anyhow::Error> {
    if let Ok(tensor_view) = tensors.tensor(name) {
        // Create a tensor of the correct shape and type
        let mut v = Tensor::zeros(
            tensor_view.shape(),
            &flash_powder::factory::TensorOptions {
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

/*


struct SafeTensorsAdaptor<'a, 's> {
    t: &'a SafeTensors<'s>,
    namespace: Vec<String>,
}

impl<'a, 's> nn::StateDictAdaptor for SafeTensorsAdaptor<'a, 's> {
    fn tensor(&self, name: &str) -> Option<Tensor> {
        safetensor_to_tensor(self.t, name).ok()
    }
    fn namespaced(&'a self, name: &str) -> SafeTensorsAdaptor<'a, 's> {
        let mut namespace = self.namespace.clone();
        namespace.push(name.to_owned());
        SafeTensorsAdaptor {
            t: self.t,
            namespace,
        }
    }
}
*/

/// Convert dynamic image into [1, 3, h, w] Tensor as floats.
fn image_to_float_tensor(
    image: &image::DynamicImage,
    use_cuda: bool,
) -> Result<Tensor, anyhow::Error> {
    let img = image.to_rgb8();

    // Lets first just tensorify the image, first create an empty tensor.
    let mut t = Tensor::zeros(
        &[img.height() as usize, img.width() as usize, 3],
        &flash_powder::factory::TensorOptions {
            dtype: Some(DType::U8),
            ..Default::default()
        },
    )?;
    // Copy in the data.
    t.data_mut()?.copy_from_slice(img.as_raw().as_slice());

    // Convert that into a float tensor and multiply it by 255.0
    let img_float = t.to(&flash_powder::factory::ToOptions {
        dtype: Some(DType::F32),
        ..Default::default()
    })?;
    let divisor: Tensor = (255.0,).try_into()?;
    let img_tensor_ready = img_float.div(&divisor)?;

    let w = img_tensor_ready.shape()[1];
    let h = img_tensor_ready.shape()[0];

    let channels_stacked = img_tensor_ready.permute(&[2, 0, 1])?;
    let with_batch = channels_stacked.view(&[1, 3, h, w])?.to_owned()?;
    let on_device = optional_cuda_to(with_batch, use_cuda)?;
    Ok(on_device)
}

pub fn main() -> Result<(), anyhow::Error> {
    use std::path::PathBuf;
    let weights = PathBuf::from("data/vgg11-8a719046.safetensors");
    if !weights.is_file() {
        eprintln!(
            "Run this binary from the 'example_vgg' directory, it looks for  \
            {}, if that doesn't exist:\n Download it from https://download.pytorch.org/models/vgg11-8a719046.pth,\
            convert it to safetensors with ./convert_pth.py",
            weights.display()
        );
        bail!("missing necessary file, bailing out")
    }

    let data = std::fs::read(weights).expect("Unable to read file");

    let tensors = SafeTensors::deserialize(&data)?;

    let use_cuda = fp::torch::cuda::is_available();
    println!("cuda available? {use_cuda:?}");

    let vgg = VGG::new(&CFG_A, &tensors, use_cuda)?;

    println!(
        "It's just label index output for now... use \
        https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/_meta.py#L7 to look them up"
    );

    for argument in std::env::args().skip(1) {
        let img = image::ImageReader::open(&argument)?.decode()?;
        let channels_stacked = image_to_float_tensor(&img, use_cuda)?;

        // println!("channels_stacked: {:?}", channels_stacked.shape());
        let r = vgg
            .forward(&channels_stacked)?
            .to(&flash_powder::factory::ToOptions {
                device: Some(Device::from_str("cpu")?),
                ..Default::default()
            })?;
        // https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/_meta.py#L7
        // Find the highest value, using some typical

        let max_item = r
            .f32s_ref()?
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        println!(
            "{argument: >50}: max_item: {max_item: >10?}, which in _meta.py#L7 is line number: {}",
            max_item.unwrap().0 + 8
        );
    }

    Ok(())
}
