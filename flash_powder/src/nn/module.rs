//! Traits for neural networks.
use anyhow;
use torch_stable::StableTorchResult;

use crate::{Ten, Tensor, core_methods::CoreMethods as _};

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

/// Value enum for [`StateDict`].
#[derive(Debug, Clone)]
pub enum Data {
    /// Tensors that record gradients, typically weights.
    Parameter(Tensor),
    /// Tensors that do not record gradients, updated during forward step.
    Buffer(Tensor),
    // State;  The non-tensor state :/
    // maybe a Box<dyn Any> ?
}
impl Data {
    pub fn as_tensor(&self) -> StableTorchResult<&Tensor> {
        match self {
            Data::Parameter(tensor) => Ok(tensor),
            Data::Buffer(tensor) => Ok(tensor),
        }
    }
    pub fn as_tensor_mut(&mut self) -> StableTorchResult<&Tensor> {
        match self {
            Data::Parameter(tensor) => Ok(tensor),
            Data::Buffer(tensor) => Ok(tensor),
        }
    }
}

/// A wrapper for the state dictionary.
#[derive(Debug, Clone, Default)]
pub struct StateDict {
    map: std::collections::HashMap<String, Data>,
}
impl StateDict {
    /// Tensors that record gradients, typically weights.
    pub fn add_parameter(&mut self, name: &str, value: Tensor) -> StableTorchResult<()> {
        self.add_data(name, Data::Parameter(value))
    }
    /// Tensors that record gradients, typically weights, only populating it if Some.
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
    /// Tensors that do not record gradients, updated during forward step.
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
    pub fn into_namespaced(mut self, ns: &str) -> Self {
        StateDict {
            map: self
                .map
                .drain()
                .map(|(k, v)| (format!("{ns}.{k}"), v))
                .collect(),
        }
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

/// Trait that specifies the input for [`Module::load_state_dict`].
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

#[derive(Debug, Default)]
pub struct ModuleTensors<'a> {
    map: std::collections::HashMap<String, &'a Tensor>,
}
impl<'a> ModuleTensors<'a> {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn insert<T: Into<String>>(&mut self, k: T, tensor: &'a Tensor) {
        let _ = self.map.insert(k.into(), tensor);
    }

    pub fn insert_optional<T: Into<String>>(&mut self, k: T, tensor: &'a Option<Tensor>) {
        if let Some(tensor) = tensor {
            let _ = self.map.insert(k.into(), tensor);
        }
    }
    pub fn insert_namespaced(&mut self, k: &str, tensors: ModuleTensors<'a>) {
        self.extend(&mut tensors.into_namespaced(&k).drain())
    }

    pub fn with<T: Into<String>>(mut self, k: T, tensor: &'a Tensor) -> Self {
        self.insert(k, tensor);
        self
    }
    pub fn with_optional<T: Into<String>>(mut self, k: T, tensor: &'a Option<Tensor>) -> Self {
        self.insert_optional(k, tensor);
        self
    }
    pub fn with_namespaced(mut self, k: &str, tensors: ModuleTensors<'a>) -> Self {
        self.extend(&mut tensors.into_namespaced(&k).drain());
        self
    }

    pub fn into_namespaced(mut self, ns: &str) -> Self {
        Self {
            map: self
                .map
                .drain()
                .map(|(k, v)| (format!("{ns}.{k}"), v))
                .collect(),
        }
    }
    pub fn drain(&mut self) -> impl Iterator<Item = (String, &'a Tensor)> {
        self.map.drain()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &&'a Tensor)> {
        self.map.iter()
    }

    pub fn extend<T: Iterator<Item = (String, &'a Tensor)>>(&mut self, t: T) {
        self.map.extend(t)
    }
}

// And the exact same with Mut :/
#[derive(Debug, Default)]
pub struct ModuleTensorsMut<'a> {
    map: std::collections::HashMap<String, &'a mut Tensor>,
}
impl<'a> ModuleTensorsMut<'a> {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn insert<T: Into<String>>(&mut self, k: T, tensor: &'a mut Tensor) {
        let _ = self.map.insert(k.into(), tensor);
    }

    pub fn insert_optional<T: Into<String>>(&mut self, k: T, tensor: &'a mut Option<Tensor>) {
        if let Some(tensor) = tensor {
            let _ = self.map.insert(k.into(), tensor);
        }
    }
    pub fn insert_namespaced(&mut self, k: &str, tensors: ModuleTensorsMut<'a>) {
        self.extend(&mut tensors.into_namespaced(&k).drain())
    }

    pub fn with<T: Into<String>>(mut self, k: T, tensor: &'a mut Tensor) -> Self {
        self.insert(k, tensor);
        self
    }
    pub fn with_optional<T: Into<String>>(mut self, k: T, tensor: &'a mut Option<Tensor>) -> Self {
        self.insert_optional(k, tensor);
        self
    }
    pub fn with_namespaced(mut self, k: &str, tensors: ModuleTensorsMut<'a>) -> Self {
        self.extend(&mut tensors.into_namespaced(&k).drain());
        self
    }

    pub fn into_namespaced(mut self, ns: &str) -> Self {
        Self {
            map: self
                .map
                .drain()
                .map(|(k, v)| (format!("{ns}.{k}"), v))
                .collect(),
        }
    }
    pub fn drain(&mut self) -> impl Iterator<Item = (String, &'a mut Tensor)> {
        self.map.drain()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &&'a mut Tensor)> {
        self.map.iter()
    }
    pub fn extend<T: Iterator<Item = (String, &'a mut Tensor)>>(&mut self, t: T) {
        self.map.extend(t)
    }
}

/// Base trait for all neural network modules.
///
/// - Pytorch Docs: <https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html>
/// - C++ docs: <https://docs.pytorch.org/cppdocs/api/nn/index.html#module-base-class>
/// - Python class code: <https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/module.py#L407>
/// - C++ class code: <https://github.com/pytorch/pytorch/blob/v2.12.0/torch/csrc/api/include/torch/nn/module.h#L63>
///
///
/// An example of how this trait looks, for a real layer like Conv2D:
///
/// ```
///  use flash_powder::{prelude::*, functional, Tensor, Ten};
///  use flash_powder::nn::module::{Module, ModuleTensors, ModuleTensorsMut};
///  #[derive(Debug, Clone)]
///  pub struct Conv2d {
///      pub weight: Tensor,
///      pub bias: Option<Tensor>,
///      pub options: functional::Conv2dOptions,
///  }
///  impl Module for Conv2d {
///      fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
///        let bias = self.bias.as_ref().map(|z| z.ten().unwrap());
///        functional::conv2d(input, &self.weight.ten()?, bias.as_ref(), &self.options)
///      }
///
///      fn tensors(&self) -> ModuleTensors<'_> {
///        ModuleTensors::new()
///           .with("weight", &self.weight)
///           .with_optional("bias", &self.bias)
///      }
///      fn tensors_mut(&mut self) -> ModuleTensorsMut<'_> {
///        ModuleTensorsMut::new()
///           .with("weight", &mut self.weight)
///           .with_optional("bias", &mut self.bias)
///      }
///  }
/// ```
///
/// Three kinds of persistent data (from c++ docs):
/// - Parameters; Tensors that record gradient, typically weights, like `weight` of linear.
/// - Buffers; Tensor that do not record gradients, typically updated during forward step; `mean`, `variance` of BatchNorm.
/// - Additionally state, not necessarily tensors, required for implementation or configuration of a Module.
///
/// <div class="warning">
///
/// If your layer has weights, don't forget to implement [`Module::tensors`] and [`Module::tensors_mut`] for the state dict
/// loading and exporting to work correctly.
///
/// </div>
///
pub trait Module: std::fmt::Debug + AsAny {
    /// Define the computation performed at every call.
    fn forward(&self, input: &Ten<'_>) -> StableTorchResult<Tensor>;
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
    /// Calls [`to`][`crate::core_methods::CoreMethods::to`] on all tensors returned by [`tensors_mut`][`Module::tensors_mut`].
    fn to(&mut self, options: &crate::factory::ToOptions) -> StableTorchResult<()> {
        for (_k, v) in self.tensors_mut().drain() {
            *v = v.to(options)?;
        }
        Ok(())
    }
    // __call__
    // __setstate__
    // __getstate__

    // state_dict
    // load_state_dict

    /// Create a state dict that hold this layer's tensors.
    ///
    /// Default implementation assumes that all tensors are weights.
    fn state_dict(&self) -> StableTorchResult<StateDict> {
        let mut d = StateDict::default();
        for (k, v) in self.tensors().drain() {
            d.add_parameter(&k, v.clone())?;
        }
        Ok(d)
    }
    /// Load a state dict into this layer.
    fn load_state_dict<'a>(&mut self, dict: &dyn StateDictReader) -> StableTorchResult<()> {
        for (k, v) in self.tensors_mut().drain() {
            *v = dict.tensor_required(&k)?;
        }
        Ok(())
    }

    fn into_boxed(self: Self) -> Box<dyn Module>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }

    /// Returns a map of references to this layer's tensors.
    ///
    /// Default implementation is an empty map, be sure to implement it if the layer has tensors.
    fn tensors(&self) -> ModuleTensors<'_> {
        Default::default()
    }

    /// Returns a map of mutable references to this layer's tensors.
    ///
    /// Default implementation is an empty map, be sure to implement it if the layer has tensors.
    fn tensors_mut(&mut self) -> ModuleTensorsMut<'_> {
        Default::default()
    }
}

// dyn_clone::clone_trait_object!(Module);

#[cfg(test)]
mod test {
    use super::super::{Conv2d, Linear, Sequential};
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
            // start.bias = None;
        }

        let seq1 = seq_root.get_mut_as::<Sequential>(1).unwrap();
        let l1: &mut Linear = seq1.get_mut_as::<Linear>(0).unwrap();
        l1.weight = zero.clone();
        // l1.bias = None;
        let l2: &mut Linear = seq1.get_mut_as::<Linear>(1).unwrap();
        l2.weight = zero.clone();
        l2.bias = Some(zero.clone());

        let seq2 = seq_root.get_mut_as::<Sequential>(2).unwrap();
        let l2: &mut Linear = seq2.get_mut_as::<Linear>(0).unwrap();
        l2.weight = zero.clone();
        l2.bias = Some(zero.clone());
        let l3: &mut Linear = seq2.get_mut_as::<Linear>(1).unwrap();
        l3.weight = zero.clone();
        // l3.bias = None;

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
