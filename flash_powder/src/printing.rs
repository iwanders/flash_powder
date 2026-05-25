//! Implements Debug and provides string conversion.
//!
//! Printing is not a light operation, and may copy tensors to be able to print them.
use crate::StableTorchResult;
use crate::core_methods::CoreMethods;
use crate::data::DataRef;
use crate::index::TensorIndex;
use crate::properties::TensorProperties;
use crate::tensor::{Ten, TenMut, Tensor, TensorAccess};

fn format_linear_tensor_at(
    t: &Ten<'_>,
    index: usize,
    options: &crate::printing::ScalarPrintOptions,
) -> StableTorchResult<String> {
    use crate::dtype::DType;
    macro_rules! generate_match {
        // Matches: an expression, then a list of (pattern, result) pairs
        ($val:expr, $(($r:ty, $p:pat)),*) => {
            match $val {
                $( $p => {
                    let v = t.d_ref::<$r>(&[index])?;
                    Ok(options.format(v))
                }, )*  // Repeatedly generate each arm
                _ => todo!("missing d_fmt for {:?}", $val),   // Optional: catch-all arm
            }
        };
    }

    generate_match!(
        t.dtype(),
        (f32, DType::F32),
        (f64, DType::F64),
        (u8, DType::U8),
        (i8, DType::I8),
        (u16, DType::U16),
        (i16, DType::I16),
        (u32, DType::U32),
        (i32, DType::I32),
        (i64, DType::I64),
        (u64, DType::U64),
        (bool, DType::Bool)
    )
}

fn format_scalar_tensor(
    t: &Ten<'_>,
    options: &crate::printing::ScalarPrintOptions,
) -> StableTorchResult<String> {
    use crate::dtype::DType;
    macro_rules! generate_match {
        // Matches: an expression, then a list of (pattern, result) pairs
        ($val:expr, $(($r:ty, $p:pat)),*) => {
            match $val {
                $( $p => {
                    let v = t.d_ref::<$r>(&[])?;
                    Ok(options.format(v))
                }, )*  // Repeatedly generate each arm
                _ => todo!("missing d_fmt for {:?}", $val),   // Optional: catch-all arm
            }
        };
    }

    generate_match!(
        t.dtype(),
        (f32, DType::F32),
        (f64, DType::F64),
        (u8, DType::U8),
        (i8, DType::I8),
        (u16, DType::U16),
        (i16, DType::I16),
        (u32, DType::U32),
        (i32, DType::I32),
        (i64, DType::I64),
        (u64, DType::U64),
        (bool, DType::Bool)
    )
}

#[derive(Copy, Clone, Debug)]
pub struct PrintOptions {
    pub precision: usize,
    pub threshold: usize,
    pub edgeitems: usize,
    pub linewidth: usize,
}
impl PrintOptions {
    pub const fn new() -> Self {
        Self {
            precision: 4,
            threshold: 1000,
            edgeitems: 3,
            linewidth: 80,
        }
    }
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            precision: 4,
            threshold: 1000,
            edgeitems: 3,
            linewidth: 80,
        }
    }
}
// And a local helper because https://github.com/rust-lang/rust/issues/118117 is not stabilised yet.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarPrintOptions {
    pub precision: Option<usize>,
    pub width: Option<usize>,
}
impl ScalarPrintOptions {
    pub fn format<T: std::fmt::Debug>(&self, value: &T) -> String {
        match (self.precision, self.width) {
            (None, None) => format!("{:?}", value),
            (None, Some(width)) => format!("{: >width$?}", value),
            (Some(precision), None) => format!("{:.precision$?}", value),
            (Some(precision), Some(width)) => format!("{: >width$.precision$?}", value),
        }
    }
}
#[derive(Copy, Clone, Debug, Default)]
pub struct TensorPrintOptions {
    pub print_options: PrintOptions,
    pub scalar_options: ScalarPrintOptions,
    pub element_width: Option<usize>,
    pub summarize: Option<bool>,
    pub indent: usize,
}

// formatter's new method is experimental... so I can't create a formatter to just test my printoptions.
use std::sync::Mutex;
static GLOBAL_PRINT_YUCK: Mutex<PrintOptions> = Mutex::new(PrintOptions::new());

/// Set the options for printing.
///
/// [pytorch equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.set_printoptions.html#torch.set_printoptions)
///
/// This writes into a global singleton that is NOT thread_local.
pub fn set_printoptions(options: &PrintOptions) {
    *(GLOBAL_PRINT_YUCK.lock().unwrap()) = *options;
}

pub trait PrintRequirements: TensorAccess + TensorProperties + CoreMethods + DataRef {}
impl<T: TensorAccess + TensorProperties + CoreMethods + DataRef + TensorIndex> PrintRequirements
    for T
{
}

// https://github.com/pytorch/pytorch/blob/8f8409cae86d725a75e2ac54ce8f93def107ced7/torch/_tensor_str.py#L130
//
// They do two passess, one to determine the width of the elements, then another to actually print.
fn tensor_format<T: PrintRequirements>(t: &T, options: &TensorPrintOptions) -> String {
    let v;
    let t = if !t.is_contiguous() {
        match t.contiguous() {
            Ok(c) => {
                v = Some(c);
                v.as_ref().map(|z| z.ten().unwrap()).unwrap()
            }
            Err(_) => return "could not make contiguous tensor".to_owned(),
        }
    } else {
        t.ten().unwrap()
    };

    let linear = t.view(&[t.numel()]).unwrap();
    let indent = options.indent;
    let summarize = options
        .summarize
        .unwrap_or(t.numel() > options.print_options.threshold);

    let determine_width = |o: &Ten<'_>, linear: &Ten<'_>| -> usize {
        let mut m = 0;
        // This is not great... but we need a contiguous tensor here to iterate over it in 1d.
        // Should we implement ravel and have it return an enum?
        // let o = (*o);
        if summarize {
            m = m.max(
                (0..options.print_options.edgeitems)
                    .map(|i| {
                        format_linear_tensor_at(&linear, i, &options.scalar_options)
                            .unwrap()
                            .len()
                    })
                    .max()
                    .unwrap_or(0),
            );
            m = m.max(
                ((o.numel() - options.print_options.edgeitems)..(o.numel()))
                    .map(|i| {
                        format_linear_tensor_at(&linear, i, &options.scalar_options)
                            .unwrap()
                            .len()
                    })
                    .max()
                    .unwrap_or(0),
            );
        } else {
            for i in 0..o.numel() {
                m = m.max(
                    format_linear_tensor_at(&linear, i, &options.scalar_options)
                        .unwrap()
                        .len(),
                )
            }
        }
        m
    };
    let mut options = *options;
    let element_width = options
        .element_width
        .unwrap_or(determine_width(&t, &linear));
    options.element_width = Some(element_width);
    // https://github.com/pytorch/pytorch/blob/8f8409cae86d725a75e2ac54ce8f93def107ced7/torch/_tensor_str.py#L242
    let vector_str = |indent: usize,
                      o: &Ten<'_>,
                      element_width: usize,
                      summarize: bool|
     -> String {
        let mut r = String::new();
        let element_length = element_width + ", ".len();
        let elements_per_line =
            ((options.print_options.linewidth - indent) / element_length).max(1);
        let mut local_options = options.scalar_options;
        local_options.width = Some(element_width);

        let mut element_str = vec![];
        if summarize {
            let left = o.narrow(0, 0, options.print_options.edgeitems).unwrap();
            element_str.extend((0..left.numel()).map(|i| {
                format_scalar_tensor(&left.i(i as isize).unwrap(), &local_options).unwrap()
            }));
            element_str.push("...".to_owned());
            let right = o
                .narrow(
                    0,
                    -(options.print_options.edgeitems as isize),
                    options.print_options.edgeitems,
                )
                .unwrap();
            element_str.extend((0..right.numel()).map(|i| {
                format_scalar_tensor(&right.i(i as isize).unwrap(), &local_options).unwrap()
            }));
        } else {
            element_str = (0..o.numel())
                .map(|i| {
                    format_scalar_tensor(&o.ten().unwrap().i(i as isize).unwrap(), &local_options)
                        .unwrap()
                })
                .collect();
        }
        let lines = element_str.chunks(elements_per_line);
        let lines: Vec<String> = lines.map(|z| z.join(", ")).collect();
        r += "[";
        let joiner = format!(",\n{}", " ".repeat(indent));
        r += &lines.join(&joiner);
        r += "]";
        r
    };
    let tensor_str = |indent: usize, o: &Ten<'_>, summarize: bool| -> String {
        let mut options = options;
        options.indent += 1;
        let mut r = String::new();
        // Okay... the big one.
        let mut slices: Vec<String> = vec![];
        // https://github.com/pytorch/pytorch/blob/8f8409cae86d725a75e2ac54ce8f93def107ced7/torch/_tensor_str.py#L304
        if summarize && t.size(0) > 2 * options.print_options.edgeitems {
            slices.extend(
                (0..options.print_options.edgeitems)
                    .map(|i| tensor_format(&crate::torch::select(&t, 0, i).unwrap(), &options)),
            );
            slices.push("...".to_owned());
            slices.extend(
                (t.size(0) - (options.print_options.edgeitems)..t.size(0))
                    .map(|i| tensor_format(&crate::torch::select(&t, 0, i).unwrap(), &options)),
            );
        } else {
            slices = (0..t.size(0))
                .map(|i| tensor_format(&crate::torch::select(&t, 0, i).unwrap(), &options))
                .collect();
        }
        r += "[";
        let joiner = format!(",{}{}", "\n".repeat(o.dim() - 1), " ".repeat(indent));
        r += &slices.join(&joiner);
        r += "]";
        r
    };

    if t.dim() == 0 {
        format_scalar_tensor(&t.ten().unwrap(), &options.scalar_options)
            .unwrap()
            .to_string()
    } else if t.dim() == 1 {
        let element_width = determine_width(&t, &linear);

        let mut r = String::new();
        r += &vector_str(indent + 1, &t, element_width, summarize);
        r
    } else {
        let mut r = String::new();
        r += &tensor_str(indent + 1, &t, summarize);
        r
    }
}

fn tensor_format_with_formatter<T: TensorAccess + TensorProperties + CoreMethods + DataRef>(
    t: &T,
    prefix: &'static str,
    fmt: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let global_options = {
        let l = GLOBAL_PRINT_YUCK.lock().unwrap();
        *l
    };
    let mut scalar_options = ScalarPrintOptions::default();
    if let Some(provided_precision) = fmt.precision() {
        scalar_options.precision = Some(provided_precision);
    } else {
        scalar_options.precision = Some(global_options.precision);
    }

    let mut r = format!("{}(", prefix);
    let print_options = TensorPrintOptions {
        print_options: global_options,
        scalar_options,
        indent: r.len(),
        summarize: None,
        element_width: None,
    };

    r += &tensor_format(&t.ten().unwrap(), &print_options);
    r += ")";
    fmt.write_fmt(format_args!("{}", r))
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        tensor_format_with_formatter(self, "Tensor", f)
    }
}

impl std::fmt::Debug for TenMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        tensor_format_with_formatter(self, "TenMut", f)
    }
}

impl std::fmt::Debug for Ten<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        tensor_format_with_formatter(self, "Ten", f)
    }
}

pub fn data_to_string<T: PrintRequirements>(v: &T, options: &TensorPrintOptions) -> String {
    tensor_format(v, options)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::index::TensorIndex;
    use crate::prelude::*;
    use crate::{StableTorchResult, Tensor};

    #[test]
    fn test_flash_powder_debug_print() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([4,4])
        */

        let d = Tensor::from(&[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ])?;
        assert_eq!(d.sizes(), &[4, 4]); // #PYTHON list(d.shape)

        // 0d
        if false {
            let sub_0d = d.i((0, 0))?;
            println!("sub_0d:\n{sub_0d:?}");
            let sub_1d = d.view(&[d.numel()])?;
            println!("sub_1d:\n{sub_1d:?}");

            println!("randn:\n{:?}", Tensor::randn(&[10], &Default::default())?);
            println!("randn:\n{:?}", Tensor::randn(&[5000], &Default::default())?);

            println!(
                "randn:\n{:?}",
                Tensor::randn(&[50, 50], &Default::default())?
            );
            println!(
                "randn:\n{:?}",
                Tensor::randn(&[50, 50, 10], &Default::default())?
            );
        }

        let d_1d = data_to_string(&d.view(&[d.numel()])?, &Default::default());
        assert_eq!(
            d_1d,
            "[ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0,\n \
               14.0, 15.0, 16.0]"
        );
        let d_2d = data_to_string(&d.view(&[4, 4])?, &Default::default());
        // println!("{d_2d}");
        assert_eq!(
            d_2d,
            "[[1.0, 2.0, 3.0, 4.0],\n \
              [5.0, 6.0, 7.0, 8.0],\n \
              [ 9.0, 10.0, 11.0, 12.0],\n \
              [13.0, 14.0, 15.0, 16.0]]"
        );
        let d_3d = data_to_string(&d.view(&[2, 2, 4])?, &Default::default());
        // println!("{d_3d}");
        assert_eq!(
            d_3d,
            "[[[1.0, 2.0, 3.0, 4.0],\n  \
            [5.0, 6.0, 7.0, 8.0]],\n\n \
            [[ 9.0, 10.0, 11.0, 12.0],\n  \
            [13.0, 14.0, 15.0, 16.0]]]"
        );

        let p = TensorPrintOptions {
            print_options: PrintOptions {
                precision: 3,
                threshold: 10,
                edgeitems: 1,
                linewidth: 80,
            },
            scalar_options: Default::default(),
            element_width: None,
            summarize: Some(true),
            indent: 0,
        };

        let d_2d = data_to_string(&d.view(&[4, 4])?, &p);
        // println!("{d_2d}");
        assert_eq!(
            d_2d,
            "[[1.0, ..., 4.0],\n \
              ...,\n \
              [13.0, ..., 16.0]]"
        );

        Ok(())
    }

    #[test]
    fn test_flash_powder_debug_print_non_contiguous() -> StableTorchResult<()> {
        let d = Tensor::from(&[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ])?;
        assert_eq!(d.sizes(), &[4, 4]);
        let r = d.i((.., 0))?;
        assert_eq!(r.is_contiguous(), false);
        let d_2d = data_to_string(&r, &Default::default());
        assert_eq!(d_2d, "[ 1.0,  5.0,  9.0, 13.0]");

        Ok(())
    }
    #[test]
    fn test_flash_powder_debug_print_multidimensional() -> StableTorchResult<()> {
        let d = Tensor::randn(&[1, 64, 448, 832], &Default::default())?;

        let d_2d = data_to_string(&d, &Default::default());
        //println!("d: {d_2d}");
        // assert_eq!(d_2d, "[ 1.0,  5.0,  9.0, 13.0]");

        Ok(())
    }
}
