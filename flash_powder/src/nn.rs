//! Torch's nn module.
//!
//! <https://docs.pytorch.org/docs/2.12/nn.html>
//!
//! This is not exposed at all through the stable API, so this is a pure rust implementation.

use anyhow;
use torch_stable::StableTorchResult;

use crate::factory::{TensorFactory, TensorOptions};
use crate::functional;
use crate::{Ten, Tensor, core_methods::CoreMethods};

// from https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=f0cee315491dc3c3b6b3f467d6a3b072
// Provide a custom trait so that we can write a blanket implementation.
pub trait AsAny {
    fn as_any_ref(&self) -> &dyn std::any::Any;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn as_any_box(self: Box<Self>) -> Box<dyn std::any::Any>;
}

impl<T> AsAny for T
where
    T: std::any::Any,
{
    // This cast cannot be written in a default implementation so cannot be
    // moved to the original trait without implementing it for every type.
    fn as_any_ref(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_any_box(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

#[derive(Debug, Clone)]
pub enum Data {
    /// Tensors that record gradients, typically weights.
    Parameter(Tensor),
    /// Tensors that do not record gradients, updated during forward step.
    Buffer(Tensor),
    // State;  The non-tensor state :/
    // maybe a Box<dyn Any> ?
}

#[derive(Debug, Clone, Default)]
pub struct StateDict {
    map: std::collections::HashMap<String, Data>,
}
impl StateDict {
    pub fn add_parameter(&mut self, name: &str, value: Tensor) -> StableTorchResult<()> {
        self.add_data(name, Data::Parameter(value))
    }
    pub fn add_optional_parameter(
        &mut self,
        name: &str,
        value: Option<Tensor>,
    ) -> StableTorchResult<()> {
        if let Some(value) = value {
            self.add_data(name, Data::Parameter(value))
        } else {
            Ok(())
        }
    }
    pub fn add_buffer(&mut self, name: &str, value: Tensor) -> StableTorchResult<()> {
        self.add_data(name, Data::Buffer(value))
    }
    pub fn add_data(&mut self, name: &str, value: Data) -> StableTorchResult<()> {
        if self.map.contains_key(name) {
            anyhow::bail!("entry with name {name} already existed")
        }
        let _ = self.map.insert(name.to_owned(), value);
        Ok(())
    }
    pub fn add_state_dict(&mut self, name: &str, mut value: StateDict) -> StableTorchResult<()> {
        // Munge paths and concat.
        for (k, v) in value.map.drain() {
            let new_name = format!("{name}.{k}");
            self.add_data(&new_name, v)?;
        }
        Ok(())
    }
    pub fn as_map(&self) -> &std::collections::HashMap<String, Data> {
        &self.map
    }
    pub fn as_map_mut(&mut self) -> &mut std::collections::HashMap<String, Data> {
        &mut self.map
    }
}

pub trait StateDictAdaptor {
    fn tensor(&self, name: &str) -> Option<Tensor>;
    fn tensor_required(&self, name: &str) -> StableTorchResult<Tensor> {
        self.tensor(name)
            .ok_or(anyhow::format_err!("missing required tensor '{name}'"))
    }
}

impl StateDictAdaptor for StateDict {
    fn tensor(&self, name: &str) -> Option<Tensor> {
        if let Some(record) = self.map.get(name) {
            match record {
                Data::Parameter(tensor) => Some(tensor.clone()),
                Data::Buffer(tensor) => Some(tensor.clone()),
            }
        } else {
            None
        }
    }
}
impl StateDictReader for StateDict {
    fn inner(&self) -> &dyn StateDictAdaptor {
        self
    }
}

pub trait StateDictReader: StateDictAdaptor {
    fn inner(&self) -> &dyn StateDictAdaptor;

    fn namespace(&self) -> Vec<String> {
        vec![]
    }

    fn namespaced<'a>(&'a self, name: &str) -> NamespacedStateDictAdaptor<'a> {
        let mut namespace = self.namespace();
        namespace.push(name.to_owned());
        NamespacedStateDictAdaptor {
            v: self.inner(),
            namespace,
        }
    }
}
pub struct NamespacedStateDictAdaptor<'a> {
    v: &'a dyn StateDictAdaptor,
    namespace: Vec<String>,
}

impl<'a> StateDictReader for NamespacedStateDictAdaptor<'a> {
    fn inner(&self) -> &'a dyn StateDictAdaptor {
        self.v
    }
    fn namespace(&self) -> Vec<String> {
        self.namespace.clone()
    }
}
impl<'a> StateDictAdaptor for NamespacedStateDictAdaptor<'a> {
    fn tensor(&self, name: &str) -> Option<Tensor> {
        let m = self.namespace.join(".") + "." + name;
        self.inner().tensor(&m)
    }
}

/// Base trait for all neural network modules.
///
/// - Pytorch Docs: <https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html>
/// - C++ docs: <https://docs.pytorch.org/cppdocs/api/nn/index.html#module-base-class>
/// - Python class code: <https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/module.py#L407>
/// - C++ class code: <https://github.com/pytorch/pytorch/blob/v2.12.0/torch/csrc/api/include/torch/nn/module.h#L63>
///
/// Three kinds of persistent data (from c++ docs):
/// - Parameters; Tensors that record gradient, typically weights, like `weight` of linear.
/// - Buffers; Tensor that do not record gradients, typically updated during forward step; `mean`, `variance` of BatchNorm.
/// - Additionally state, not necessarily tensors, required for implementation or configuration of a Module.
pub trait Module: std::fmt::Debug + AsAny {
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error>;
    // These look relevant;
    // register_buffer
    // register_parameter
    // add_module / register_module
    // get_submodule
    // set_submodule
    // get_parameter
    // get_buffer
    // get_extra_state
    // set_extra_state
    // apply
    // to
    // __call__
    // __setstate__
    // __getstate__
    // state_dict
    // load_state_dict
    fn state_dict(&self) -> StableTorchResult<StateDict> {
        Ok(Default::default())
    }
    fn load_state_dict<'a>(&mut self, dict: &dyn StateDictReader) -> StableTorchResult<()> {
        let _ = dict;
        Ok(())
    }

    fn into_boxed(self: Self) -> Box<dyn Module>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

// dyn_clone::clone_trait_object!(Module);

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
            return Ok(input.to_owned()?);
        }
        let mut intermediate = self.modules.first().unwrap().forward(input)?;
        for remaining_layers in self.modules.iter().skip(1) {
            intermediate = remaining_layers.forward(&intermediate.ten()?)?;
        }
        Ok(intermediate)
    }
    fn state_dict(&self) -> StableTorchResult<StateDict> {
        let mut m: StateDict = Default::default();
        for (i, v) in self.modules.iter().enumerate() {
            let name = format!("{i}");
            m.add_state_dict(&name, v.state_dict()?)?
        }
        Ok(m)
    }
    fn load_state_dict(&mut self, dict: &dyn StateDictReader) -> StableTorchResult<()> {
        for (i, v) in self.modules.iter_mut().enumerate() {
            let name = format!("{i}");
            v.load_state_dict(&dict.namespaced(&name))?
        }
        Ok(())
    }
}
impl Sequential {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn push<T: Into<Box<dyn Module>>>(&mut self, m: T) {
        self.modules.push(m.into())
    }
    pub fn push_boxed(&mut self, m: Box<dyn Module>) {
        self.modules.push(m)
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Box<dyn Module>> {
        self.modules.get_mut(index)
    }
    pub fn get_mut_as<T: Sized + 'static>(&mut self, index: usize) -> Option<&mut T> {
        use std::ops::DerefMut;
        if let Some(v) = self.get_mut(index) {
            v.deref_mut().as_any_mut().downcast_mut()
        } else {
            None
        }
    }
    pub fn get(&self, index: usize) -> Option<&Box<dyn Module>> {
        self.modules.get(index)
    }
    pub fn get_as<T: Sized + 'static>(&self, index: usize) -> Option<&T> {
        use std::ops::Deref;
        if let Some(v) = self.get(index) {
            v.deref().as_any_ref().downcast_ref()
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
            items.push(item.into_boxed());
        }

        items
    }
}

impl FromIterator<Box<dyn Module>> for Sequential {
    fn from_iter<I: IntoIterator<Item = Box<dyn Module>>>(iter: I) -> Self {
        let mut items = Sequential::new();

        for item in iter {
            items.push(item);
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
    fn state_dict(&self) -> StableTorchResult<StateDict> {
        let mut m: StateDict = Default::default();
        m.add_parameter("weight", self.weight.clone())?;
        m.add_optional_parameter("bias", self.bias.clone())?;
        Ok(m)
    }
    fn load_state_dict(&mut self, dict: &dyn StateDictReader) -> StableTorchResult<()> {
        self.weight = dict.tensor_required("weight")?;
        self.bias = dict.tensor("bias");
        Ok(())
    }
}
impl Conv2d {
    /// A new conv2d layer, without bias.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (i64, i64),
        options: functional::Conv2dOptions,
    ) -> StableTorchResult<Conv2d> {
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
        Ok(Self {
            weight,
            bias: None,
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

/// Helper to read a linear layer from the safetensors.
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
    fn state_dict(&self) -> StableTorchResult<StateDict> {
        let mut m: StateDict = Default::default();
        m.add_parameter("weight", self.weight.clone())?;
        m.add_optional_parameter("bias", self.bias.clone())?;
        Ok(m)
    }
    fn load_state_dict(&mut self, dict: &dyn StateDictReader) -> StableTorchResult<()> {
        self.weight = dict.tensor_required("weight")?;
        self.bias = dict.tensor("bias");
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_state_dict() -> StableTorchResult<()> {
        let mut s = StateDict::default();
        let one: Tensor = (1.0,).try_into()?;
        let two: Tensor = (2.0,).try_into()?;
        let three: Tensor = (3.0,).try_into()?;
        s.add_buffer("foo", one.clone())?;
        assert_eq!(s.add_buffer("foo", one.clone()).is_err(), true);
        s.add_buffer("foo.bar", two.clone())?;
        s.add_buffer("foo.bar.buz", three.clone())?;

        assert_eq!(s.tensor("foo").unwrap().as_f64()?, &1.0);
        assert_eq!(s.tensor("foo.bar").unwrap().as_f64()?, &2.0);
        assert_eq!(s.tensor("foo.bar.buz").unwrap().as_f64()?, &3.0);

        let foo = s.namespaced("foo");
        assert_eq!(foo.tensor("bar").unwrap().as_f64()?, &2.0);
        assert_eq!(foo.tensor("bar.buz").unwrap().as_f64()?, &3.0);

        let bar = foo.namespaced("bar");
        assert_eq!(bar.tensor("buz").unwrap().as_f64()?, &3.0);
        Ok(())
    }

    #[test]
    fn test_flash_powder_save_load() -> StableTorchResult<()> {
        let zero: Tensor = (0.0,).try_into()?;
        let one: Tensor = (1.0,).try_into()?;
        let two: Tensor = (2.0,).try_into()?;
        let three: Tensor = (3.0,).try_into()?;
        let four: Tensor = (4.0,).try_into()?;

        let linear1 = Linear {
            weight: one,
            bias: None,
        };
        let linear2 = Linear {
            weight: two.clone(),
            bias: Some(two.clone()),
        };
        let linear3 = Linear {
            weight: three.clone(),
            bias: Some(three.clone()),
        };
        let seq: Sequential = vec![linear1, linear2.clone()].drain(..).collect();
        let seq2: Sequential = vec![linear2.into_boxed(), linear3.into_boxed()]
            .drain(..)
            .collect();

        let mut seq_root: Sequential = Default::default();
        seq_root.push_boxed(Box::new(Conv2d {
            weight: three.clone(),
            bias: Some(four.clone()),
            options: Default::default(),
        }));
        seq_root.push_boxed(Box::new(seq));
        seq_root.push_boxed(Box::new(seq2));
        println!("root: {seq_root:#?}");
        let seq_root_at_start = format!("{:?}", seq_root);

        let s = seq_root.state_dict()?;
        println!("s: {s:?}");

        // THis works, but it's hardly a test.
        seq_root.load_state_dict(&s)?;

        // let z = seq_root.get_mut(0).unwrap();
        // This is quite the mouth ful :/
        // let convmut: &mut Conv2d = (*z).deref_mut().as_any_mut().downcast_mut().unwrap();
        // Okay; this then;
        {
            let _: &mut Conv2d = seq_root.get_mut_as::<Conv2d>(0).unwrap();
            let _: &Conv2d = seq_root.get_as::<Conv2d>(0).unwrap();
        }
        // Now that we can get mutable access into the sequential... after it was created, we can zero out its
        // tensors lol.
        {
            let start: &mut Conv2d = seq_root.get_mut_as::<Conv2d>(0).unwrap();
            start.weight = zero.clone();
            start.bias = None;
        }

        let seq1 = seq_root.get_mut_as::<Sequential>(1).unwrap();
        let l1: &mut Linear = seq1.get_mut_as::<Linear>(0).unwrap();
        l1.weight = zero.clone();
        l1.bias = None;
        let l2: &mut Linear = seq1.get_mut_as::<Linear>(1).unwrap();
        l2.weight = zero.clone();
        l2.bias = None;

        let seq2 = seq_root.get_mut_as::<Sequential>(2).unwrap();
        let l2: &mut Linear = seq2.get_mut_as::<Linear>(0).unwrap();
        l2.weight = zero.clone();
        l2.bias = None;
        let l3: &mut Linear = seq2.get_mut_as::<Linear>(1).unwrap();
        l3.weight = zero.clone();
        l3.bias = None;

        // Now that should all be zero'd out.
        println!("root: {seq_root:?}");
        assert_ne!(format!("{:?}", seq_root), seq_root_at_start);

        // Now load the state dict.
        seq_root.load_state_dict(&s)?;

        // Now it should be identical to the start, that confirms we have loaded all tensors again.
        assert_eq!(format!("{:?}", seq_root), seq_root_at_start);

        Ok(())
    }
}
