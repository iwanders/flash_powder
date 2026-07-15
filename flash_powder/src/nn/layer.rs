//! Layers implementing [`Module`].
//!
//! These are prety much structs that wrap functions in [`crate::functional`] but also own the tensor weights.
use super::module::{Module, ModuleTensors, ModuleTensorsMut};
use crate::factory::TensorOptions;
use crate::functional;
use crate::prelude::*;
use crate::{StableTorchResult, Ten, Tensor};

/// Sequential module
///
/// - pytorch equivalent; <https://docs.pytorch.org/docs/2.12/generated/torch.nn.Sequential.html>
#[derive(Debug, Default)]
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}
impl Module for Sequential {
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        if self.modules.is_empty() {
            return input.to_owned();
        }
        let mut intermediate = self.modules.first().unwrap().forward(input)?;
        for remaining_layers in self.modules.iter().skip(1) {
            intermediate = remaining_layers.forward(&intermediate.ten()?)?;
        }
        Ok(intermediate)
    }

    fn tensors(&self) -> ModuleTensors<'_> {
        let mut res: ModuleTensors = Default::default();
        for (i, sublayer) in self.modules.iter().enumerate() {
            res.insert_namespaced(&format!("{i}"), sublayer.tensors())
        }
        res
    }
    fn tensors_mut(&mut self) -> ModuleTensorsMut<'_> {
        let mut res: ModuleTensorsMut = Default::default();
        for (i, sublayer) in self.modules.iter_mut().enumerate() {
            res.insert_namespaced(&format!("{i}"), sublayer.tensors_mut())
        }
        res
    }
}
impl Sequential {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn push<T: Sized + 'static + Module>(&mut self, m: T) {
        let b: Box<dyn Module> = Box::new(m);
        self.modules.push(b)
    }
    pub fn push_boxed(&mut self, m: Box<dyn Module>) {
        self.modules.push(m)
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut dyn Module> {
        if let Some(boxed_value) = self.modules.get_mut(index) {
            Some(&mut (**boxed_value))
        } else {
            None
        }
    }
    pub fn get_mut_as<T: Sized + 'static>(&mut self, index: usize) -> Option<&mut T> {
        if let Some(v) = self.get_mut(index) {
            (v as &mut dyn std::any::Any).downcast_mut()
        } else {
            None
        }
    }
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.modules.get(index).map(|z| &**z)
    }
    pub fn get_as<T: Sized + 'static>(&self, index: usize) -> Option<&T> {
        if let Some(v) = self.get(index) {
            (v as &dyn std::any::Any).downcast_ref()
        } else {
            None
        }
    }
}

use std::iter::FromIterator;

impl<T: Module + 'static> FromIterator<T> for Sequential {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut items = Sequential::new();

        for item in iter {
            items.push(item);
        }

        items
    }
}

impl FromIterator<Box<dyn Module>> for Sequential {
    fn from_iter<I: IntoIterator<Item = Box<dyn Module>>>(iter: I) -> Self {
        let mut items = Sequential::new();

        for item in iter {
            items.push_boxed(item);
        }

        items
    }
}

/// Conv2d
///
/// - pytorch equivalent; <https://docs.pytorch.org/docs/2.12/generated/torch.nn.Conv2d.html>
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub options: functional::Conv2dOptions,
}
impl Module for Conv2d {
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        let bias = self.bias.as_ref().map(|z| z.ten().unwrap());
        functional::conv2d(input, &self.weight.ten()?, bias.as_ref(), &self.options)
    }

    fn tensors(&self) -> ModuleTensors<'_> {
        ModuleTensors::new()
            .with("weight", &self.weight)
            .with_optional("bias", &self.bias)
    }
    fn tensors_mut(&mut self) -> ModuleTensorsMut<'_> {
        ModuleTensorsMut::new()
            .with("weight", &mut self.weight)
            .with_optional("bias", &mut self.bias)
    }
}
impl Conv2d {
    /// A new conv2d layer, with bias.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (i64, i64),
        options: functional::Conv2dOptions,
    ) -> StableTorchResult<Self> {
        // Weights is; [out_channels, in_channels / groups, kernel_size[0], kernel_size[1]]
        let channels_div_groups = ((in_channels as i64) / options.groups) as usize;
        let weight = Tensor::zeros(
            &[
                out_channels,
                channels_div_groups,
                kernel_size.0 as usize,
                kernel_size.1 as usize,
            ],
            &TensorOptions::default(),
        )?;
        // Bias is; [out_channels].
        let bias = Tensor::zeros(&[out_channels], &TensorOptions::default())?;
        Ok(Self {
            weight,
            bias: Some(bias),
            options,
        })
    }
}

/// ConvTranspose2d
///
/// - pytorch equivalent; <https://docs.pytorch.org/docs/2.12/generated/torch.nn.ConvTranspose2d.html>
#[derive(Debug, Clone)]
pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub options: functional::ConvTranspose2dOptions,
}
impl Module for ConvTranspose2d {
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        let bias = self.bias.as_ref().map(|z| z.ten().unwrap());
        functional::conv_transpose2d(input, &self.weight.ten()?, bias.as_ref(), &self.options)
    }

    fn tensors(&self) -> ModuleTensors<'_> {
        ModuleTensors::new()
            .with("weight", &self.weight)
            .with_optional("bias", &self.bias)
    }
    fn tensors_mut(&mut self) -> ModuleTensorsMut<'_> {
        ModuleTensorsMut::new()
            .with("weight", &mut self.weight)
            .with_optional("bias", &mut self.bias)
    }
}
impl ConvTranspose2d {
    /// A new conv2d layer, with bias.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (i64, i64),
        options: functional::ConvTranspose2dOptions,
    ) -> StableTorchResult<Self> {
        // Weights is; [out_channels, in_channels / groups, kernel_size[0], kernel_size[1]]
        let channels_div_groups = ((in_channels as i64) / options.groups) as usize;
        let weight = Tensor::zeros(
            &[
                out_channels,
                channels_div_groups,
                kernel_size.0 as usize,
                kernel_size.1 as usize,
            ],
            &TensorOptions::default(),
        )?;
        // Bias is; [out_channels].
        let bias = Tensor::zeros(&[out_channels], &TensorOptions::default())?;
        Ok(Self {
            weight,
            bias: Some(bias),
            options,
        })
    }
}

/// Relu
///
/// - pytorch equivalent; <https://docs.pytorch.org/docs/2.12/generated/torch.nn.ReLU.html>
#[derive(Debug, Clone)]
pub struct ReLU;
impl Module for ReLU {
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        functional::relu(input)
    }
}

/// Maxpool2D
///
/// - pytorch equivalent; <https://docs.pytorch.org/docs/2.12/generated/torch.nn.ReLU.html>
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    pub kernel_size: (i64, i64),
    pub options: functional::MaxPool2dDOptions,
}
impl Module for MaxPool2d {
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        functional::max_pool2d(input, self.kernel_size, &self.options)
    }
}

/// Linear
///
/// - pytorch equivalent; <https://docs.pytorch.org/docs/2.12/generated/torch.nn.Linear.html>
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}
impl Module for Linear {
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        let bias = self.bias.as_ref().map(|z| z.ten().unwrap());
        functional::linear(input, &self.weight.ten()?, bias.as_ref())
    }
    fn tensors(&self) -> ModuleTensors<'_> {
        ModuleTensors::new()
            .with("weight", &self.weight)
            .with_optional("bias", &self.bias)
    }
    fn tensors_mut(&mut self) -> ModuleTensorsMut<'_> {
        ModuleTensorsMut::new()
            .with("weight", &mut self.weight)
            .with_optional("bias", &mut self.bias)
    }
}
impl Linear {
    /// A new linear layer, without bias.
    pub fn new_without_bias(in_features: usize, out_features: usize) -> StableTorchResult<Self> {
        // Weights is; [out_features,in_features]
        let weight = Tensor::zeros(&[out_features, in_features], &TensorOptions::default())?;
        Ok(Self { weight, bias: None })
    }
    /// A new linear layer, with bias.
    pub fn new(in_features: usize, out_features: usize) -> StableTorchResult<Self> {
        // Weights is; [out_features,in_features]
        let weight = Tensor::zeros(&[out_features, in_features], &TensorOptions::default())?;
        // Bias is; [out_features]
        let bias = Some(Tensor::zeros(&[out_features], &TensorOptions::default())?);
        Ok(Self { weight, bias })
    }
}

/// A placeholder identity operator
///
/// This can be very useful fo replace layers that do nothing in inference to keep the layer indices identical.
#[derive(Debug, Clone)]
pub struct Identity;

impl Module for Identity {
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        input.to_owned()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_layer_sequential_cast() {
        let mut v = Sequential::new();
        v.push(Identity);
        let r = v.get_as::<Identity>(0);
        assert!(r.is_some());
        let r = v.get_mut_as::<Identity>(0);
        assert!(r.is_some());
    }
}
