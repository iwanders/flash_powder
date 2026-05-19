//! Torch's nn module.
//!
//! <https://docs.pytorch.org/docs/2.12/nn.html>
//!
//! This is not exposed at all through the stable API, so this is a pure rust implementation.

use anyhow;
use torch_stable::StableTorchResult;

use crate::functional;
use crate::{Ten, Tensor, core_methods::CoreMethods};

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
    fn namespaced(&self, name: &str) -> NamespacedStateDictAdaptor<'_>;
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
    fn namespaced(&self, name: &str) -> NamespacedStateDictAdaptor<'_> {
        NamespacedStateDictAdaptor {
            state_dict: self,
            prefix: name.to_owned(),
        }
    }
}
pub struct NamespacedStateDictAdaptor<'a> {
    state_dict: &'a dyn StateDictAdaptor,
    prefix: String,
}
impl<'a> StateDictAdaptor for NamespacedStateDictAdaptor<'a> {
    fn tensor(&self, name: &str) -> Option<Tensor> {
        self.state_dict.tensor(&format!("{}.{}", self.prefix, name))
    }
    fn namespaced(&self, name: &str) -> NamespacedStateDictAdaptor<'_> {
        NamespacedStateDictAdaptor {
            state_dict: self,
            prefix: name.to_owned(),
        }
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
pub trait Module: std::fmt::Debug + dyn_clone::DynClone {
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
    fn load_state_dict(&mut self, dict: &dyn StateDictAdaptor) -> StableTorchResult<()> {
        let _ = dict;
        Ok(())
    }
}

dyn_clone::clone_trait_object!(Module);

/// Sequential module
///
/// - pytorch equivalent; <https://docs.pytorch.org/docs/2.12/generated/torch.nn.Sequential.html>
#[derive(Debug, Clone)]
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
    fn load_state_dict(&mut self, dict: &dyn StateDictAdaptor) -> StableTorchResult<()> {
        for (i, v) in self.modules.iter_mut().enumerate() {
            let name = format!("{i}");
            v.load_state_dict(&dict.namespaced(&name))?
        }
        Ok(())
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
    fn load_state_dict(&mut self, dict: &dyn StateDictAdaptor) -> StableTorchResult<()> {
        self.weight = dict.tensor_required("weight")?;
        self.bias = dict.tensor("bias");
        Ok(())
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
    fn load_state_dict(&mut self, dict: &dyn StateDictAdaptor) -> StableTorchResult<()> {
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
}
