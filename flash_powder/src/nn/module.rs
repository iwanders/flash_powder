//! Traits for neural networks.
use anyhow;
use torch_stable::StableTorchResult;

use crate::{
    Ten, Tensor,
    core_methods::{CoreMethods as _, CoreMethodsMut},
};

pub use std::collections::HashSet;

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
    fn ten<'d>(&'d self, name: &str) -> Option<Ten<'d>>;
    fn ten_required<'d>(&'d self, name: &str) -> StableTorchResult<Ten<'d>> {
        self.ten(name)
            .ok_or(anyhow::format_err!("missing required tensor '{name}'"))
    }
    fn keys(&self) -> HashSet<String>;
}

impl StateDictAdaptor for StateDict {
    fn ten<'d>(&'d self, name: &str) -> Option<Ten<'d>> {
        if let Some(record) = self.map.get(name) {
            match record {
                Data::Parameter(tensor) => tensor.ten().ok(),
                Data::Buffer(tensor) => tensor.ten().ok(),
            }
        } else {
            None
        }
    }
    fn keys(&self) -> HashSet<String> {
        self.map.keys().map(|a| a.to_owned()).collect()
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
    fn ten<'d>(&'d self, name: &str) -> Option<Ten<'d>> {
        let m = self.namespace.join(".") + "." + name;
        self.inner().ten(&m)
    }

    fn keys(&self) -> HashSet<String> {
        let prefix = self.namespace.join(".") + ".";
        let prefix = &prefix;
        self.inner()
            .keys()
            .iter()
            .filter_map(|a| {
                if a.starts_with(prefix) {
                    Some(a.replace(prefix, ""))
                } else {
                    None
                }
            })
            .collect()
    }
}

// Private type to encapsulate tensors present in the module.
#[derive(Debug)]
enum ModuleTensor<'a> {
    Always(&'a Tensor),
    Optional(&'a Option<Tensor>),
}
impl<'a> From<&'a Tensor> for ModuleTensor<'a> {
    fn from(value: &'a Tensor) -> Self {
        ModuleTensor::Always(value)
    }
}
impl<'a> From<&'a Option<Tensor>> for ModuleTensor<'a> {
    fn from(value: &'a Option<Tensor>) -> Self {
        ModuleTensor::Optional(value)
    }
}

#[derive(Debug, Default)]
pub struct ModuleTensors<'a> {
    map: std::collections::HashMap<String, ModuleTensor<'a>>,
}
impl<'a> ModuleTensors<'a> {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn insert<T: Into<String>>(&mut self, k: T, tensor: &'a Tensor) {
        let _ = self.map.insert(k.into(), tensor.into());
    }

    pub fn insert_optional<T: Into<String>>(&mut self, k: T, tensor: &'a Option<Tensor>) {
        let _ = self.map.insert(k.into(), tensor.into());
    }
    pub fn insert_namespaced(&mut self, k: &str, tensors: ModuleTensors<'a>) {
        self.extend(&mut tensors.into_namespaced(k).drain())
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
        self.extend(&mut tensors.into_namespaced(k).drain());
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
        self.map.drain().filter_map(|(k, v)| match v {
            ModuleTensor::Always(tensor) => Some((k, tensor)),
            ModuleTensor::Optional(optional_tensor) => optional_tensor.as_ref().map(|a| (k, a)),
        })
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &'a Tensor)> {
        self.map.iter().filter_map(|(k, v)| match v {
            ModuleTensor::Always(tensor) => Some((k, *tensor)),
            ModuleTensor::Optional(optional_tensor) => optional_tensor.as_ref().map(|a| (k, a)),
        })
    }

    pub fn extend<T: Iterator<Item = (String, &'a Tensor)>>(&mut self, t: T) {
        self.map.extend(t.map(|(k, v)| (k, v.into())))
    }
}

// Private type to encapsulate tensors present in the module.
#[derive(Debug)]
enum ModuleTensorMut<'a> {
    Always(&'a mut Tensor),
    Optional(&'a mut Option<Tensor>),
}
impl<'a> From<&'a mut Tensor> for ModuleTensorMut<'a> {
    fn from(value: &'a mut Tensor) -> Self {
        ModuleTensorMut::Always(value)
    }
}
impl<'a> From<&'a mut Option<Tensor>> for ModuleTensorMut<'a> {
    fn from(value: &'a mut Option<Tensor>) -> Self {
        ModuleTensorMut::Optional(value)
    }
}
impl<'a> ModuleTensorMut<'a> {
    fn as_option(&'a mut self) -> Option<&'a mut Tensor> {
        match self {
            ModuleTensorMut::Always(tensor) => Some(tensor),
            ModuleTensorMut::Optional(t) => t.as_mut(),
        }
    }
}

// And the exact same with Mut :/

#[derive(Debug, Default)]
pub struct ModuleTensorsMut<'a> {
    map: std::collections::HashMap<String, ModuleTensorMut<'a>>,
}
impl<'a> ModuleTensorsMut<'a> {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn insert<T: Into<String>>(&mut self, k: T, tensor: &'a mut Tensor) {
        let _ = self.map.insert(k.into(), tensor.into());
    }

    pub fn insert_optional<T: Into<String>>(&mut self, k: T, tensor: &'a mut Option<Tensor>) {
        let _ = self.map.insert(k.into(), tensor.into());
    }
    pub fn insert_namespaced(&mut self, k: &str, tensors: ModuleTensorsMut<'a>) {
        self.extend(&mut tensors.into_namespaced(k).drain())
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
        self.extend(&mut tensors.into_namespaced(k).drain());
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
        self.map.drain().filter_map(|(k, v)| match v {
            ModuleTensorMut::Always(tensor) => Some((k, tensor)),
            ModuleTensorMut::Optional(optional_tensor) => optional_tensor.as_mut().map(|a| (k, a)),
        })
    }
    pub fn iter(&'a mut self) -> impl Iterator<Item = (&'a String, &'a mut Tensor)> {
        self.map.iter_mut().filter_map(move |(k, v)| {
            if let Some(z) = v.as_option() {
                Some((k, z))
            } else {
                None
            }
        })
    }

    /// Retrieve the optional into which the tensor by name may be stored.
    ///
    /// This allows clearing the tensor.
    pub fn get_optional<'b>(&'b mut self, name: &str) -> Option<&'b mut Option<Tensor>> {
        if let Some(value) = self.map.get_mut(name) {
            match value {
                ModuleTensorMut::Always(_) => None,
                ModuleTensorMut::Optional(t) => Some(t),
            }
        } else {
            None
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    pub fn extend<T: Iterator<Item = (String, &'a mut Tensor)>>(&mut self, t: T) {
        self.map.extend(t.map(|(k, v)| (k, v.into())))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct StateDictLoadOptions {
    /// State dictionary keys must strictly be equivalent to the Module keys.
    ///
    /// Identical to the python functionality.
    pub strict: bool,

    /// Assign the tensors in the module instead of copying the data into them.
    ///
    /// This wipes the tensor properties in the module completely and uses those from the state dict.
    ///
    /// Identical to the python functionality.
    pub assign: bool,

    /// Clear optional tensors if present in the destination but not in the state dictionary.
    ///
    /// This has no equivalent on the python side.
    pub clear_optional: bool,

    /// Populate optional tensors if not yet populated in the destination and present in the state dictionary.
    ///
    /// This always assigns, regardless of the [`assign`] field.
    ///
    /// This has no equivalent on the python side.
    pub populate_optional: bool,
}

impl Default for StateDictLoadOptions {
    fn default() -> Self {
        Self {
            strict: true,
            assign: false,
            clear_optional: false,
            populate_optional: false,
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
///
/// An example of how this trait looks, for a real layer like Conv2D:
///
/// ```
///  use flash_powder::{prelude::*, nn::functional, Tensor, Ten};
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
pub trait Module: std::fmt::Debug + std::any::Any {
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
    /*
     Notes from the python side:
        Loading a state dict raises if the tensor sizes don't match exactly.
        Loading a state dict raises if any tensor is present that was not expected.
        Loading a state dict raises for any tensor that is expected but missing.

        In short, it is super strict.

        That's less than ideal, because tensor size having to match is more annoying than just having to match the topology
        with the python side.

        Maybe:
        struct LoadOptions{
            pub check_tensor_size: bool,
            pub allow_unused_tensors: bool,
        }

        That allows enforcing the tensor size.
        And allow_unused_tensors = false captures situations where the rust side has Option<Tensor>'s with None, that are in the dict?
        It doesn't allow clearing Options though.... maybe clear_optional: bool

         https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
         strict (bool, optional) – whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict() function. Default: True
         assign (bool, optional) – When set to False, the properties of the tensors in the current module are preserved whereas setting it to True preserves properties of the Tensors in the state dict.

        but that doesn't handle the current signatures...
        Do we also want to be able to inject metadata? It can be convenient if the model topology can be in the safetensors metadata.
    */
    fn load_state_dict(
        &mut self,
        dict: &dyn StateDictReader,
        options: &StateDictLoadOptions,
    ) -> StableTorchResult<()> {
        if options.strict {
            let existing_keys = dict.keys();
            let desired_keys: HashSet<String> =
                self.tensors().iter().map(|(a, _t)| a.clone()).collect();
            let too_many_keys: HashSet<&String> = existing_keys.difference(&desired_keys).collect();
            if !too_many_keys.is_empty() {
                anyhow::bail!("strict mode enabled got too many keys: {too_many_keys:?}");
            }
        }

        if options.clear_optional || options.populate_optional {
            let keys: Vec<String> = self.tensors_mut().keys().cloned().collect();
            // Retrieve the optional keys, check if it exists, if not clear them, then fall through to the assign.
            let mut tensors_mut = self.tensors_mut();
            for k in keys {
                // Only operate on tensors marked optional.
                if let Some(optional_tensor) = tensors_mut.get_optional(&k) {
                    // See if this exists in the dict.
                    if let Some(ten_in_dict) = dict.ten(&k) {
                        // It is not present in the destination, but assign is true, so we can populate the optional.
                        if optional_tensor.is_none() && options.populate_optional {
                            // This may lead to an extra assignment in the assignment for loop below.
                            *optional_tensor = Some(ten_in_dict.to_owned()?);
                        }
                    } else {
                        // Tensor not present in the state dictionary
                        if options.clear_optional {
                            // Optional tensor is not present in the dictionary, so we clear it.
                            *optional_tensor = None
                        }
                    }
                }
            }
        }

        for (k, v) in self.tensors_mut().drain() {
            if options.assign {
                // Assign the tensor directly, overwriting properties.
                *v = dict.ten_required(&k)?.to_owned()?;
            } else {
                // Copy from the tensor, keeping its properties.
                v.copy_from_tensor(&dict.ten_required(&k)?)?
            }
        }

        Ok(())
    }

    fn into_boxed(self) -> Box<dyn Module>
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
    use crate::{DType, prelude::*};

    #[test]
    fn test_flash_powder_state_dict() -> StableTorchResult<()> {
        let mut s = StateDict::default();
        let one: Tensor = 1.0.try_into()?;
        let two: Tensor = 2.0.try_into()?;
        let three: Tensor = 3.0.try_into()?;
        s.add_buffer("foo", one.clone())?;
        assert_eq!(s.add_buffer("foo", one.clone()).is_err(), true);
        s.add_buffer("foo.bar", two.clone())?;
        s.add_buffer("foo.bar.buz", three.clone())?;

        assert_eq!(
            s.keys(),
            HashSet::<String>::from([
                "foo".to_owned(),
                "foo.bar".to_owned(),
                "foo.bar.buz".to_owned()
            ])
        );

        // assert_eq!(s.tensor("foo").unwrap().as_f64()?, &1.0);
        assert_eq!(s.ten("foo").unwrap().as_f64()?, &1.0);
        // assert_eq!(s.tensor("foo.bar").unwrap().as_f64()?, &2.0);
        assert_eq!(s.ten("foo.bar").unwrap().as_f64()?, &2.0);
        // assert_eq!(s.tensor("foo.bar.buz").unwrap().as_f64()?, &3.0);
        assert_eq!(s.ten("foo.bar.buz").unwrap().as_f64()?, &3.0);

        let foo = s.namespaced("foo");
        // assert_eq!(foo.tensor("bar").unwrap().as_f64()?, &2.0);
        assert_eq!(foo.ten("bar").unwrap().as_f64()?, &2.0);
        // assert_eq!(foo.tensor("bar.buz").unwrap().as_f64()?, &3.0);
        assert_eq!(foo.ten("bar.buz").unwrap().as_f64()?, &3.0);
        assert_eq!(
            foo.keys(),
            HashSet::<String>::from(["bar".to_owned(), "bar.buz".to_owned()])
        );

        let bar = foo.namespaced("bar");
        // assert_eq!(bar.tensor("buz").unwrap().as_f64()?, &3.0);
        assert_eq!(bar.ten("buz").unwrap().as_f64()?, &3.0);
        assert_eq!(bar.keys(), HashSet::<String>::from(["buz".to_owned()]));
        Ok(())
    }

    #[test]
    fn test_flash_powder_save_load() -> StableTorchResult<()> {
        let zero: Tensor = 0.0.try_into()?;
        let one: Tensor = 1.0.try_into()?;
        let two: Tensor = 2.0.try_into()?;
        let three: Tensor = 3.0.try_into()?;
        let four: Tensor = 4.0.try_into()?;

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
        seq_root.load_state_dict(&s, &Default::default())?;

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

        seq_root.load_state_dict(&s, &Default::default())?;

        // Now it should be identical to the start, that confirms we have loaded all tensors again.
        assert_eq!(format!("{:?}", seq_root), seq_root_at_start);

        Ok(())
    }

    #[test]
    fn test_flash_powder_save_load_options() -> StableTorchResult<()> {
        let f64_one: Tensor = 0.0.try_into()?;
        assert_eq!(f64_one.dtype(), DType::F64);

        let f32_2: Tensor = 2.0f32.try_into()?;
        assert_eq!(f32_2.dtype(), DType::F32);

        {
            //  strict enabled, one key too many in source state dict

            let linear1 = Linear {
                weight: f64_one.clone(),
                bias: Some(f64_one.clone()),
            };
            let l1_dict = linear1.state_dict()?;

            let mut linear2 = Linear {
                weight: f32_2.clone(),
                bias: None,
            };
            let options = StateDictLoadOptions {
                strict: true,
                assign: false,
                clear_optional: false,
                populate_optional: false,
            };
            let r = linear2.load_state_dict(&l1_dict, &options);
            assert!(r.is_err());
            assert_eq!(
                format!("{:?}", r.err().unwrap()),
                "strict mode enabled got too many keys: {\"bias\"}"
            );
        }

        {
            //  strict enabled, one key missing in source, this is a normal 'missing required tensor'.

            let linear1 = Linear {
                weight: f64_one.clone(),
                bias: None,
            };
            let l1_dict = linear1.state_dict()?;

            let mut linear2 = Linear {
                weight: f32_2.clone(),
                bias: Some(f64_one.clone()),
            };
            let options = StateDictLoadOptions {
                strict: true,
                assign: false,
                clear_optional: false,
                populate_optional: false,
            };
            let r = linear2.load_state_dict(&l1_dict, &options);
            assert!(r.is_err());

            assert_eq!(
                format!("{:?}", r.err().unwrap()),
                "missing required tensor 'bias'"
            );
        }

        {
            // Next, assign equals false means it should keep properties and copy values.

            let linear1 = Linear {
                weight: f64_one.clone(),
                bias: None,
            };
            let l1_dict = linear1.state_dict()?;

            let mut linear2 = Linear {
                weight: f32_2.clone(),
                bias: None,
            };
            let options = StateDictLoadOptions {
                strict: false,
                assign: false,
                clear_optional: false,
                populate_optional: false,
            };
            linear2.load_state_dict(&l1_dict, &options)?;
            assert_eq!(linear2.weight.dtype(), DType::F32);
        }

        {
            // Next, assign equals true means the original source tensor properties are blown away.

            let linear1 = Linear {
                weight: f64_one.clone(),
                bias: None,
            };
            let l1_dict = linear1.state_dict()?;

            let mut linear2 = Linear {
                weight: f32_2.clone(),
                bias: None,
            };
            let options = StateDictLoadOptions {
                strict: false,
                assign: true,
                clear_optional: false,
                populate_optional: false,
            };
            linear2.load_state_dict(&l1_dict, &options)?;
            assert_eq!(linear2.weight.dtype(), DType::F64);
        }

        if true {
            // Finally, check if we can clear optionals if they're not present.
            let mut linear1 = Linear {
                weight: f64_one.clone(),
                bias: None,
            };
            let l1_dict = linear1.state_dict()?;

            let mut linear2 = Linear {
                weight: f32_2.clone(),
                bias: Some(f64_one.clone()),
            };
            let l2_dict = linear2.state_dict()?;
            let options = StateDictLoadOptions {
                strict: false,
                assign: false,
                clear_optional: true,
                populate_optional: false,
            };
            // This situation clears the optioanl.
            linear2.load_state_dict(&l1_dict, &options)?;
            assert!(linear2.bias.is_none());

            // With the config from above, populate_optional is false, so we can't assign into a None.
            linear1.load_state_dict(&l2_dict, &options)?;
            assert!(linear1.bias.is_none());

            // If we change that to populating optionals, it should assign;
            let options = StateDictLoadOptions {
                strict: false,
                assign: true,
                clear_optional: false,
                populate_optional: true,
            };
            linear1.load_state_dict(&l2_dict, &options)?;
            assert!(linear1.bias.is_some());
        }
        Ok(())
    }
}
