//! Quickly thrown together example of VGG11
//!
//! `https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html#torchvision.models.vgg11`
//! Code:
//! `https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L91`.
//!
//
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py

// Smallest is vgg11;
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L306-L329
// which is of the 'A' category.
use anyhow::bail;

use flash_powder as fp;
use flash_powder::{Ten, Tensor, functional, nn, prelude::*};
use nn::module::{Module, ModuleTensors, ModuleTensorsMut};

// -------------- VGG Implementation --------------
// The config, as per https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L91
// but then as integers.
const CFG_A: &[u32] = &[
    64, 'M' as u32, 128, 'M' as u32, 256, 256, 'M' as u32, 512, 512, 'M' as u32, 512, 512,
    'M' as u32,
];

#[derive(Debug)]
/// VGG struct with layers and classifier.
pub struct VGG {
    features: nn::Sequential,
    classifier: nn::Sequential,
}
impl VGG {
    /// Create a new vgg config as per the layer specification and the provided weights.
    pub fn new(features: nn::Sequential) -> Result<Self, anyhow::Error> {
        // https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L43
        // and then the group called classifier
        let mut classifier: nn::Sequential = Default::default();
        classifier.push(nn::Linear::new(512 * 7 * 7, 4096)?);
        classifier.push(nn::ReLU);
        classifier.push(nn::Identity); // actually a dropout, here to keep the indexing for state dict correct.

        // next linear block
        classifier.push(nn::Linear::new(4096, 4096)?);
        classifier.push(nn::ReLU);
        classifier.push(nn::Identity);

        // last linear.
        const NUM_CLASSESS: usize = 1000;
        classifier.push(nn::Linear::new(4096, NUM_CLASSESS)?);

        Ok(VGG {
            features,
            classifier,
        })
    }
}
impl nn::Module for VGG {
    // https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L65
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        let mut r = self.features.forward(input)?;
        r = functional::adaptive_avg_pool2d(&r, (7, 7))?;
        r = r.flatten(1, None)?;
        r = self.classifier.forward(&r.ten()?)?;

        Ok(r)
    }
    fn tensors(&self) -> ModuleTensors<'_> {
        ModuleTensors::new()
            .with_namespaced("features", self.features.tensors())
            .with_namespaced("classifier", self.classifier.tensors())
    }

    fn tensors_mut(&mut self) -> ModuleTensorsMut<'_> {
        ModuleTensorsMut::new()
            .with_namespaced("features", self.features.tensors_mut())
            .with_namespaced("classifier", self.classifier.tensors_mut())
    }
}

fn make_layers(cfg: &[u32]) -> Result<nn::Sequential, anyhow::Error> {
    // https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L73
    let mut features: nn::Sequential = Default::default();
    let mut in_channels = 3;
    for v in cfg.iter() {
        if *v == 'M' as u32 {
            // 'M' denotes maxpool layer.
            features.push(nn::MaxPool2d {
                kernel_size: (2, 2),
                options: Default::default(),
            });
        } else {
            let conv2d_options = functional::Conv2dOptions {
                stride: (1, 1),
                padding: (1, 1),
                ..Default::default()
            };
            let layer = nn::Conv2d::new(in_channels, *v as usize, (3, 3), conv2d_options)?;
            features.push(layer);
            features.push(nn::ReLU);
            in_channels = *v as usize;
        }
    }
    Ok(features)
}

// -------------- image::DynamicImage to fp::Tensor --------------
/// Convert dynamic image into [1, 3, h, w] Tensor as floats.
fn image_to_float_tensor(
    image: &image::DynamicImage,
    use_cuda: bool,
) -> Result<fp::Tensor, anyhow::Error> {
    let img = image.to_rgb8();

    // Lets first just tensorify the image, first create an empty tensor.
    let mut t = fp::Tensor::zeros(
        &[img.height() as usize, img.width() as usize, 3],
        &fp::factory::TensorOptions {
            dtype: Some(fp::DType::U8),
            ..Default::default()
        },
    )?;
    // Copy in the data.
    t.data_mut()?.copy_from_slice(img.as_raw().as_slice());

    // Convert that into a float tensor and multiply it by 255.0
    let img_float = t.to(&fp::factory::ToOptions {
        dtype: Some(fp::DType::F32),
        ..Default::default()
    })?;
    let divisor: Tensor = 255.0.try_into()?;
    let img_tensor_ready = img_float.div(&divisor)?;

    let w = img_tensor_ready.shape()[1];
    let h = img_tensor_ready.shape()[0];

    let channels_stacked = img_tensor_ready.permute(&[2, 0, 1])?;
    let with_batch = channels_stacked.view(&[1, 3, h, w])?.to_owned()?;
    if use_cuda {
        Ok(with_batch.to(&fp::Device::CUDA.into())?)
    } else {
        Ok(with_batch)
    }
}

pub fn main() -> Result<(), anyhow::Error> {
    use std::path::PathBuf;

    // Verify weights exist, if not give a nice warning.
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

    // Load safetensors and wrap
    let data = std::fs::read(weights).expect("Unable to read file");
    let tensors = flash_powder_safetensors::safetensors::SafeTensors::deserialize(&data)?;
    let reader = flash_powder_safetensors::SafetensorReader::from_safetensors(&tensors);

    // Instantiate vgg network and load its weights.
    let features = make_layers(CFG_A)?;
    let mut vgg = VGG::new(features)?;
    vgg.load_state_dict(&reader, &Default::default())?;

    // Move to cuda if available.
    let use_cuda = fp::torch::cuda::is_available();
    println!("cuda available? {use_cuda:?}");
    if use_cuda {
        vgg.to(&fp::Device::CUDA.into())?
    }

    // Print how to interpret the returned value.
    println!(
        "It's just (label index, value) output for now... use \
        https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/_meta.py#L7 to look them up"
    );

    // Iterate over the input arguments and run the network.
    for argument in std::env::args().skip(1) {
        let img = image::ImageReader::open(&argument)?.decode()?;
        let channels_stacked = image_to_float_tensor(&img, use_cuda)?;

        let r = vgg
            .forward(&channels_stacked.ten()?)?
            .to(&flash_powder::factory::ToOptions {
                device: Some(fp::Device::CPU),
                ..Default::default()
            })?;
        // https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/_meta.py#L7
        const INDEX_TO_LINE_NUMBER: usize = 8;

        // Find the highest value.
        let max_item = r
            .f32s_ref()?
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        println!(
            "{argument: >50}: max_item: {max_item: >10?}, which in _meta.py#L7 is line number: {}",
            max_item.unwrap().0 + INDEX_TO_LINE_NUMBER
        );
    }

    Ok(())
}
