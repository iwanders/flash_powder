//! Traits for neural networks.
use anyhow;
use torch_stable::StableTorchResult;

use crate::{Ten, Tensor};

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
    // __call__
    // __setstate__
    // __getstate__
    // state_dict
    // load_state_dict
    /// Create a state dict that hold this layer's tensors.
    fn state_dict(&self) -> StableTorchResult<StateDict> {
        Ok(Default::default())
    }
    /// Load a state dict into this layer.
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
