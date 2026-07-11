//! Helper tooling for interop with [image].
//!
//! Main functionality:
//! - [`TensorToImage::save_image`] to write an image to disk from a tensor.
//! - [`TensorFromImage::read_image`] to read an image from disk into a `[C, H, W]` tensor.
//! - [`ImageToTensor::to_tensor`] worker to convert image types to tensor, used by [`TensorFromImage::read_image`].
//! - [`TensorImageOperations`] common operations, like converting `[0u8, 255]` to `[0.0f32, 1.0]`.
//!

use flash_powder as fp;
use fp::Tensor;

use anyhow::bail;
use flash_powder::prelude::*;
pub use image;
use zerocopy;

use fp::StableTorchResult;

pub mod prelude {
    pub use super::{ImageToTensor, TensorFromImage, TensorImageOperations, TensorToImage};
}

pub trait TensorToImage {
    /// Save this tensor as an image.
    ///
    /// Semantics are based on size and data type, for greyscale it requires an explicit channel dimension to ensure it
    /// can be distinguished from the batch.
    /// - `[H, W]`: This is a single greyscale image.
    /// - `[1, H, W]`: This is a single greyscale image.
    /// - `[3, 1, H, W]`: This is three greyscale images.
    /// - `[2, 3, 1, H, W]`: This is six greyscale images.
    /// - `[3, H, W]`: This is an RGB image.
    /// - `[1, 3, H, W]`: This is an RGB image.
    /// - `[2, 3, H, W]`: This is two RGB images.
    /// - `[2, 2, 3, H, W]`: This is four RGB images.
    ///
    /// In the above description, 3 can be replaced with 4 to make it RGBA.
    ///
    /// The interleaved flavour (`[H, W, 3]`) is not handled, neither are 2 channel images.
    ///
    /// - All integer types are expected to fit within a byte `[0, 255]`.
    /// - All (supported) float types in `[0.0, 1.0]`.
    /// - All images are exported as `[0, 255]` u8.
    ///
    /// The pytorch side only accepts (B x C x H x W), with an argument to specify number per row, this function puts
    /// the B dimension always on the same row, but you can do  (V x B x C x H x W), where V is stacking rows.
    ///
    /// This calls into [`Self::to_dynamic_image`], and then calls [`image::DynamicImage::save`].
    fn save_image<Q>(&self, path: Q) -> StableTorchResult<()>
    where
        Q: AsRef<std::path::Path>;

    /// Converts to a dynamic image, follows the exact same semantics as save_image.
    fn to_dynamic_image(&self) -> StableTorchResult<image::DynamicImage>;
}

#[derive(Debug)]
struct DTypeInfo {
    bytes_per_pixel: usize,
    is_integer: bool,
    is_float: bool,
}
impl DTypeInfo {
    fn s(bytes_per_pixel: usize, is_integer: bool, is_float: bool) -> Self {
        Self {
            bytes_per_pixel,
            is_integer,
            is_float,
        }
    }
}
fn get_info(d: fp::DType) -> DTypeInfo {
    match d {
        flash_powder::DType::U8 => DTypeInfo::s(1, true, false),
        flash_powder::DType::I8 => DTypeInfo::s(1, true, false),
        flash_powder::DType::I16 => DTypeInfo::s(2, true, false),
        flash_powder::DType::I32 => DTypeInfo::s(4, true, false),
        flash_powder::DType::I64 => DTypeInfo::s(8, true, false),
        flash_powder::DType::U16 => DTypeInfo::s(2, true, false),
        flash_powder::DType::U32 => DTypeInfo::s(4, true, false),
        flash_powder::DType::U64 => DTypeInfo::s(8, true, false),
        flash_powder::DType::F16 => DTypeInfo::s(2, false, true),
        flash_powder::DType::F32 => DTypeInfo::s(4, false, true),
        flash_powder::DType::F64 => DTypeInfo::s(8, false, true),
        _ => DTypeInfo::s(0, false, false),
    }
}

impl<T> TensorToImage for T
where
    T: TensorProperties + DataRef + CoreMethods,
{
    fn to_dynamic_image(&self) -> StableTorchResult<image::DynamicImage> {
        // - `[H, W]`: This is a single greyscale image.
        // - `[1, H, W]`: This is a single greyscale image.
        // - `[3, 1, H, W]`: This is three greyscale images.
        // - `[2, 3, 1, H, W]`: This is six greyscale images.
        // - `[3, H, W]`: This is an RGB image.
        // - `[1, 3, H, W]`: This is an RGB image.
        // - `[2, 3, H, W]`: This is two RGB images.
        // - `[2, 2, 3, H, W]`: This is four RGB images.
        // - `[4, H, W]`: This is an RGBA image.
        // - `[1, 4, H, W]`: This is an RGBA image.
        // - `[2, 4, H, W]`: This is two RGBA images.
        // - `[2, 2, 4, H, W]`: This is four RGBA images.
        let greyscale = self.dim() == 2 || self.dim() >= 3 && self.isize(-3) == 1;
        let rgb = self.dim() >= 3 && self.isize(-3) == 3;
        let rgba = self.dim() >= 3 && self.isize(-3) == 4;
        if !greyscale && !rgb && !rgba {
            bail!(
                "could not infer rgb(a) or greyscale, tensor size was: {:?}",
                self.shape()
            );
        }

        // If we got the first flavour, that [H, W], we have a dimension less and permute is non-generic, so first
        // we make this generic on three dimensions.
        let v = if self.dim() == 2 {
            self.unsqueeze(0)?
        } else {
            self.ten()?
        };
        let v = match v.dim() {
            3 => v.unsqueeze(0)?.unsqueeze(0)?,
            4 => v.unsqueeze(0)?,
            5 => v,
            _ => unreachable!(),
        };
        assert_eq!(v.dim(), 5);
        // The dimension is now consistent on [V, B, C, H, W]

        // Negative indices because at first the dim count wasn't constant.
        let image_width = v.isize(-1);
        let image_height = v.isize(-2);
        let image_per_row = v.isize(-4);
        let image_rows = v.isize(-5);
        // Perform a grandiose swap to interleave the data and ensure the batch grid is correct.

        // The dimension is now consistent on [V, B, C, H, W]
        //                                     0  1  2  3  4
        // We need to permute again...
        // V, B, H, W, 3
        // But with B=2, that results in images looking like:
        // AA AA
        // BB BB
        //
        // Instead of
        // AA BB
        // AA BB
        // Need to swap B & H
        // V H B W 3
        //
        // and the channel swap, so we go to;
        // [V, H, B, W, 3]
        let v = v.permute(&[0, 3, 1, 4, 2])?.contiguous()?;

        let info = get_info(v.dtype());

        let v = if info.is_float {
            // Lets do this calculation in F32 space to be precise?
            let vf32 = v.to(&fp::DType::F32.into())?;
            let s_f32: Tensor = 255.0.try_into()?;
            // Multiply by 255.0, then convert to u8.
            vf32.mul(&s_f32)?.to(&fp::DType::U8.into())?
        } else if info.is_integer {
            // We expect this is already correctly scaled, so just normalise to u8.
            v.to(&fp::DType::U8.into())?
        } else {
            bail!("image is not of a supported dtype")
        };

        if info.bytes_per_pixel == 0 {
            bail!("unsupported dtype: {:?}", v.dtype())
        }

        // Next, we can make the dynamic image.
        let width = (image_per_row * image_width) as u32;
        let height = (image_rows * image_height) as u32;

        //let mut img = image::DynamicImage::new(out_width as u32, out_height as u32, color_type);
        // And now it should be a single byte copy? O_o
        // Save for the fact that as_mut_bytes() doesn't exist... so we need to actually handle them seperately.
        if greyscale {
            let mut g = image::GrayImage::new(width, height);
            g.as_flat_samples_mut()
                .as_mut_slice()
                .copy_from_slice(v.data()?);
            Ok(image::DynamicImage::ImageLuma8(g))
        } else if rgb {
            let mut g = image::RgbImage::new(width, height);
            g.as_flat_samples_mut()
                .as_mut_slice()
                .copy_from_slice(v.data()?);
            Ok(image::DynamicImage::ImageRgb8(g))
        } else if rgba {
            let mut g = image::RgbaImage::new(width, height);
            g.as_flat_samples_mut()
                .as_mut_slice()
                .copy_from_slice(v.data()?);
            Ok(image::DynamicImage::ImageRgba8(g))
        } else {
            unreachable!("unhandled image channel count")
        }
    }

    fn save_image<Q>(&self, path: Q) -> StableTorchResult<()>
    where
        Q: AsRef<std::path::Path>,
    {
        let dynamic = self.to_dynamic_image()?;
        dynamic.save(path).map_err(Into::into)
    }
}

pub trait ImageToTensor {
    /// Convert the image to a tensor.
    ///
    /// - Implemented for [`image::DynamicImage`].
    ///
    /// The values of the output tensor are in [`fp::DType::U8`] in `[0, 255]` for most cases, it's shape is `[C, H, W]`.
    ///
    /// Images always become 3 dimensional;
    /// - greyscale: `[1, H, W]`.
    /// - rgb:  `[3, H, W]`.
    /// - rgba: `[4, H, W]`.
    ///
    /// Other channel counts are currently not supported.
    ///
    /// Type is chosen based on data width per channel:
    /// - 1 byte: [`fp::DType::U8`]
    /// - 2 byte: [`fp::DType::U16`]
    /// - 4 byte: [`fp::DType::F32`]
    ///
    /// Similar to TorchVision's Like [decode_image](https://docs.pytorch.org/vision/main/generated/torchvision.io.decode_image.html#torchvision.io.decode_image).
    ///
    /// `output (Tensor[image_channels, image_height, image_width])`
    ///
    ///
    /// This follows the exact same semantics as [`TensorFromImage::read_image`].
    fn to_tensor(&self) -> StableTorchResult<Tensor>;
}

impl ImageToTensor for image::DynamicImage {
    fn to_tensor(&self) -> StableTorchResult<Tensor> {
        let color = self.color();
        let channels = color.channel_count() as usize;
        let bytes_per_pixel = color.bytes_per_pixel() as usize;
        let width = self.width() as usize;
        let height = self.height() as usize;

        let dtype = match bytes_per_pixel / channels {
            1 => fp::DType::U8,
            2 => fp::DType::U16,
            4 => fp::DType::F32, // An image buffer for 32-bit float RGB pixels
            _ => bail!(
                "unhandled input bytes_per_pixel {bytes_per_pixel} / channels {channels} should be  1, 2 or 4"
            ),
        };

        // Create an empty tensor.
        let mut t = fp::Tensor::zeros(
            &[height, width, channels],
            &fp::factory::TensorOptions {
                dtype: Some(dtype),
                ..Default::default()
            },
        )?;

        // Copy in the data.
        t.data_mut()?.copy_from_slice(self.as_bytes());

        // And finally, perform the channel swap.
        let channels_stacked = t.permute(&[2, 0, 1])?;

        // ANd return an owned version.
        channels_stacked.to_owned()
    }
}

pub trait TensorFromImage {
    /// Read an image from disk.
    ///
    /// This loads an [`image::DynamicImage`] from disk and calls [`ImageToTensor::to_tensor`] on it.
    fn read_image<Q>(path: Q) -> StableTorchResult<Tensor>
    where
        Q: AsRef<std::path::Path>;
}

impl TensorFromImage for Tensor {
    fn read_image<Q>(path: Q) -> StableTorchResult<Tensor>
    where
        Q: AsRef<std::path::Path>,
    {
        let img = image::ImageReader::open(path)?.decode()?;
        img.to_tensor()
    }
}

pub trait TensorImageOperations {
    /// Convert an image from integer space to floats in `[0.0, 1.0]`.
    ///
    /// If the [`ToOptions::dtype`][`fp::factory::ToOptions::dtype`] field is empty, it will use [`DType::F32`][`fp::DType::F32`].
    /// Calculation happens in this type, as well as being the return type, for `[0u8, 255]` using [`F16`][`fp::DType::F16`] does not result in loss of precision.
    ///
    /// Integers are scaled with their maximum value.
    fn image_floatify(&self, options: &fp::factory::ToOptions) -> StableTorchResult<Tensor>;

    /// Scales a tensor's values to fit within `[0.0, 1.0]`.
    ///
    /// This is helpful if you need to visualise a tensor.
    ///
    /// This does:
    /// ```nocode
    /// span = self.max() - self.min()
    /// out = (self - self.min()) / span
    /// ```
    ///
    fn image_scale_to_domain(&self) -> StableTorchResult<Tensor>;
}
impl TensorImageOperations for Tensor {
    fn image_floatify(&self, options: &fp::factory::ToOptions) -> StableTorchResult<Tensor> {
        self.ten()?.image_floatify(options)
    }
    fn image_scale_to_domain(&self) -> StableTorchResult<Tensor> {
        self.ten()?.image_scale_to_domain()
    }
}
impl<'a> TensorImageOperations for fp::Ten<'a> {
    fn image_floatify(&self, options: &fp::factory::ToOptions) -> StableTorchResult<Tensor> {
        let calc_type = options.dtype.unwrap_or(fp::DType::F32);
        let desired_device = options.device.unwrap_or(self.device());
        let divisor: Tensor = if self.dtype() == fp::DType::U8 {
            (u8::MAX).try_into()?
        } else if self.dtype() == fp::DType::U16 {
            (u16::MAX).try_into()?
        } else if self.dtype() == fp::DType::U32 {
            (u32::MAX).try_into()?
        } else {
            1.0.try_into()?
        };
        // Move the image and divisor to the desired device.
        let to_options = fp::factory::ToOptions {
            dtype: Some(calc_type),
            device: Some(desired_device),
            ..Default::default()
        };
        let image = self.to(&to_options)?;
        let divisor = divisor.to(&to_options)?;
        image.div(&divisor)
    }

    fn image_scale_to_domain(&self) -> StableTorchResult<Tensor> {
        let min = self.min()?;
        let max = self.max()?;
        let span = max.sub(&min)?;
        self.sub(&min)?.div(&span)
    }
}

pub trait FlatSamplesToTensor<Buffer> {
    fn as_ten<'a, T>(&'a self) -> StableTorchResult<fp::Ten<'a>>
    where
        Buffer: AsRef<[T]>,
        T: zerocopy::IntoBytes + zerocopy::Immutable + 'a,
        T: fp::dtype::ScalarDType;

    // fn to_tensor(&self) -> StableTorchResult<Tensor> {
    //     self.as_ten()?.to_owned()
    // }
}
impl<Buffer> FlatSamplesToTensor<Buffer> for image::flat::FlatSamples<Buffer> {
    fn as_ten<'a, T>(&'a self) -> StableTorchResult<flash_powder::Ten<'a>>
    where
        Buffer: AsRef<[T]>,
        T: zerocopy::IntoBytes + zerocopy::Immutable + 'a,
        T: fp::dtype::ScalarDType,
    {
        use image::flat::NormalForm;
        // The `ÌmageBuffer` uses row major form with packed samples.
        // ImagePacked is C, H, W
        // RowMajorPacked is H, W, C (without gaps in C)

        let image_buffer_reqs = self.layout.is_normal(NormalForm::PixelPacked)
            && self.layout.is_normal(NormalForm::RowMajorPacked);

        if image_buffer_reqs {
            // This is H, W, C.
            let sizes = &[
                self.layout.height as _,
                self.layout.width as _,
                self.layout.channels as _,
            ];
            let height_stride = (self.layout.channels as usize)
                .checked_mul(self.layout.width as usize)
                .ok_or(anyhow::format_err!("too big"))?;
            let strides = &[height_stride as _, self.layout.channels as _, 1];
            let dtype = T::type_dtype();
            dbg!(sizes);
            dbg!(strides);

            let options = fp::tensor::BlobOptionsBytes {
                sizes,
                strides,
                dtype,
            };
            let slice: &'a [T] = self.samples.as_ref();

            // Example utilizing zerocopy to read bytes safely
            let byte_slice: &'a [u8] = zerocopy::IntoBytes::as_bytes(slice);
            fp::Ten::from_bytes(byte_slice, &options)
        } else {
            bail!(" unsupported layout {:?}", self.layout);
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_read_write_image() -> StableTorchResult<()> {
        let u8_255: Tensor = 255u8.try_into()?;
        let f32_255: Tensor = 255.0f32.try_into()?;
        let u16_255: Tensor = 255u16.try_into()?;

        // Float, 6 by 6 pixel of greyscale, top left quadrant set to white.
        let mut d = Tensor::zeros(&[6, 6], &Default::default())?;
        d.i_mut((0..3, 0..3))?.fill_f64(1.0)?;
        d.save_image("/tmp/fp_greyscale_f32.png").unwrap();
        let img = image::ImageReader::open(&"/tmp/fp_greyscale_f32.png")?.decode()?;
        assert!(matches!(img, image::DynamicImage::ImageLuma8(_)));
        println!("d: {d:?}");
        let floatified = img.to_tensor()?.image_floatify(&Default::default())?;
        let floatified = floatified.squeeze()?;
        println!("img: {img:?}");
        println!("floatified: {floatified:?}");
        assert!(d.is_equal(&floatified)?);
        let img = img.to_luma8();
        assert_eq!(img.get_pixel(0, 0), &image::Luma([255]));
        assert_eq!(img.get_pixel(5, 0), &image::Luma([0]));
        let v = Tensor::read_image("/tmp/fp_greyscale_f32.png")?.to(&fp::DType::U8.into())?;
        assert!(d.unsqueeze(0)?.mul(&f32_255)?.is_equal(&v)?);

        // U8, 6 by 6 pixel of greyscale, top left quadrant set to white.
        let mut d = Tensor::zeros(&[6, 6], &fp::DType::U8.into())?;
        d.i_mut((0..3, 0..3))?.fill_tensor(&u8_255)?;
        d.save_image("/tmp/fp_greyscale_u8.png").unwrap();
        let img = image::ImageReader::open(&"/tmp/fp_greyscale_u8.png")?.decode()?;
        assert!(matches!(img, image::DynamicImage::ImageLuma8(_)));
        let img = img.to_luma8();
        assert_eq!(img.get_pixel(0, 0), &image::Luma([255]));
        assert_eq!(img.get_pixel(5, 0), &image::Luma([0]));
        let v = Tensor::read_image("/tmp/fp_greyscale_u8.png")?.to(&fp::DType::U8.into())?;
        assert!(d.unsqueeze(0)?.is_equal(&v)?);

        // U16, 6 by 6 pixel of greyscale, top left quadrant set to white.
        let mut d = Tensor::zeros(&[6, 6], &fp::DType::U16.into())?;
        d.i_mut((0..3, 0..3))?.fill_tensor(&u16_255)?;
        d.save_image("/tmp/fp_greyscale_u16.png").unwrap();
        let img = image::ImageReader::open(&"/tmp/fp_greyscale_u16.png")?.decode()?;
        assert!(matches!(img, image::DynamicImage::ImageLuma8(_)));
        let v = Tensor::read_image("/tmp/fp_greyscale_u16.png")?.to(&fp::DType::U16.into())?;
        assert!(d.unsqueeze(0)?.is_equal(&v)?);
        let img = img.to_luma8();
        assert_eq!(img.get_pixel(0, 0), &image::Luma([255]));
        assert_eq!(img.get_pixel(5, 0), &image::Luma([0]));

        // Test an RGB image.
        let mut d = Tensor::zeros(&[3, 6, 6], &Default::default())?;
        // Top left, R
        d.i_mut((0, 0..3, 0..3))?.fill_f64(1.0)?;
        // Bottom left, G
        d.i_mut((1, 3..6, 0..3))?.fill_f64(1.0)?;
        // Top right Blue
        d.i_mut((2, 0..3, 3..6))?.fill_f64(1.0)?;
        // Bottom right, white
        d.i_mut((.., 3..6, 3..6))?.fill_f64(1.0)?;
        d.save_image("/tmp/fp_rgb_f32.png").unwrap();
        let img = image::ImageReader::open(&"/tmp/fp_rgb_f32.png")?.decode()?;
        assert!(matches!(img, image::DynamicImage::ImageRgb8(_)));
        let v = Tensor::read_image("/tmp/fp_rgb_f32.png")?
            .to(&fp::DType::F32.into())?
            .div(&f32_255)?;
        assert!(d.is_equal(&v)?);
        assert!(d.is_equal(&img.to_tensor()?.image_floatify(&Default::default())?)?);

        let img = img.to_rgb8();
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
        assert_eq!(img.get_pixel(5, 0), &image::Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(0, 5), &image::Rgb([0, 255, 0]));
        assert_eq!(img.get_pixel(5, 5), &image::Rgb([255, 255, 255]));

        // Test an rgba image.
        let mut d = Tensor::zeros(&[4, 6, 6], &Default::default())?;
        // Top left, R
        d.i_mut((0, 0..3, 0..3))?.fill_f64(1.0)?;
        // Bottom left, G
        d.i_mut((1, 3..6, 0..3))?.fill_f64(1.0)?;
        // Top right Blue
        d.i_mut((2, 0..3, 3..6))?.fill_f64(1.0)?;
        // Bottom right, white, this also sets the full opacity.
        d.i_mut((.., 3..6, 3..6))?.fill_f64(1.0)?;
        // Opacity for the middle section.
        d.i_mut((3, 1..5, 1..5))?.fill_f64(1.0)?;
        d.save_image("/tmp/fp_rgba_f32.png")?;
        let img = image::ImageReader::open(&"/tmp/fp_rgba_f32.png")?.decode()?;
        assert!(matches!(img, image::DynamicImage::ImageRgba8(_)));
        let v = Tensor::read_image("/tmp/fp_rgba_f32.png")?
            .to(&fp::DType::F32.into())?
            .div(&f32_255)?;
        assert!(d.is_equal(&v)?);
        assert!(d.is_equal(&img.to_tensor()?.image_floatify(&Default::default())?)?);

        let img = img.to_rgba8();
        // Transparent ones on the borders
        assert_eq!(img.get_pixel(0, 0), &image::Rgba([255, 0, 0, 0]));
        assert_eq!(img.get_pixel(5, 0), &image::Rgba([0, 0, 255, 0]));
        assert_eq!(img.get_pixel(0, 5), &image::Rgba([0, 255, 0, 0]));
        // White section bottom right.
        assert_eq!(img.get_pixel(5, 5), &image::Rgba([255, 255, 255, 255]));
        // Opaque centers
        assert_eq!(img.get_pixel(1, 1), &image::Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(4, 1), &image::Rgba([0, 0, 255, 255]));
        assert_eq!(img.get_pixel(2, 4), &image::Rgba([0, 255, 0, 255]));

        // And if we add a zero dimension into the channel, we should get 4, 1, 6, 6, so four images in a batch.
        let batched = d.unsqueeze(1)?;
        batched.save_image("/tmp/fp_rgba_f32_batch.png")?;

        // These batches we can't really test with is_equals... since we make a composite.
        let mut d = Tensor::zeros(&[3, 3, 6, 6], &Default::default())?;
        d.i_mut((0, 0, 0..6, 0..6))?.fill_f64(1.0)?; // first image in batch red
        d.i_mut((1, 2, 0..6, 0..6))?.fill_f64(1.0)?; // second image in batch blue.
        d.i_mut((2, 1, 0..6, 0..6))?.fill_f64(1.0)?; // third image in batch green.
        d.save_image("/tmp/fp_rgb_b2.png").unwrap();
        let img = image::ImageReader::open(&"/tmp/fp_rgb_b2.png")?.decode()?;
        assert!(matches!(img, image::DynamicImage::ImageRgb8(_)));
        let back = img.to_tensor()?;
        let square_255 = Tensor::ones(&[6, 6], &fp::DType::U8.into())?.mul(&u8_255)?;
        let square_0 = Tensor::zeros(&[6, 6], &fp::DType::U8.into())?.mul(&u8_255)?;
        // First square;
        assert!(back.i((0, 0..6, 0..6))?.is_equal(&square_255)?);
        assert!(back.i((1, 0..6, 0..6))?.is_equal(&square_0)?);
        assert!(back.i((2, 0..6, 0..6))?.is_equal(&square_0)?);
        let stacked = fp::torch::stack(&[&square_255, &square_0, &square_0], 0)?;
        assert!(back.i((.., 0..6, 0..6))?.is_equal(&stacked)?);
        // Second square is blue.
        let stacked = fp::torch::stack(&[&square_0, &square_0, &square_255], 0)?;
        assert!(back.i((.., 0..6, 6..12))?.is_equal(&stacked)?);
        // Last square is green
        let stacked = fp::torch::stack(&[&square_0, &square_255, &square_0], 0)?;
        assert!(back.i((.., 0..6, 12..18))?.is_equal(&stacked)?);

        // Now test the row functionality.
        let mut d = Tensor::zeros(&[2, 3, 3, 6, 6], &Default::default())?;
        d.i_mut((0, 0, 0, 0..6, 0..6))?.fill_f64(1.0)?; // first image in 1st batch red
        d.i_mut((0, 1, 1..3, 0..6, 0..6))?.fill_f64(1.0)?; // second image in 1st batch cyan.
        d.i_mut((0, 2, 1, 0..6, 0..6))?.fill_f64(1.0)?; // third image in 1st batch green.
        d.i_mut((1, 0, 2, 0..6, 0..6))?.fill_f64(1.0)?; // first image in 2nd batch blue
        d.i_mut((1, 1, 2, 0..6, 0..6))?.fill_f64(1.0)?; // second image in 2nd batch blue.
        d.i_mut((1, 2, 0..2, 0..6, 0..6))?.fill_f64(1.0)?; // third image in 2nd batch yellow
        d.save_image("/tmp/fp_rgb_2r_b2.png").unwrap();

        let img = image::ImageReader::open(&"/tmp/fp_rgb_2r_b2.png")?.decode()?;
        assert!(matches!(img, image::DynamicImage::ImageRgb8(_)));
        let back = img.to_tensor()?;
        // First square is red.
        let stacked = fp::torch::stack(&[&square_255, &square_0, &square_0], 0)?;
        assert!(back.i((.., 0..6, 0..6))?.is_equal(&stacked)?);
        // Second square is cyan.
        let stacked = fp::torch::stack(&[&square_0, &square_255, &square_255], 0)?;
        assert!(back.i((.., 0..6, 6..12))?.is_equal(&stacked)?);
        // Last square is green
        let stacked = fp::torch::stack(&[&square_0, &square_255, &square_0], 0)?;
        assert!(back.i((.., 0..6, 12..18))?.is_equal(&stacked)?);
        // Second row first two square is blue;
        let stacked = fp::torch::stack(&[&square_0, &square_0, &square_255], 0)?;
        assert!(back.i((.., 6..12, 0..6))?.is_equal(&stacked)?);
        assert!(back.i((.., 6..12, 6..12))?.is_equal(&stacked)?);
        // and the last one is yellow.
        let stacked = fp::torch::stack(&[&square_255, &square_255, &square_0], 0)?;
        assert!(back.i((.., 6..12, 12..18))?.is_equal(&stacked)?);

        Ok(())
    }

    #[test]
    fn test_image_floatify_f16() -> StableTorchResult<()> {
        for i in 0..255u8 {
            let t: Tensor = i.try_into()?;
            let floatified_default = t.image_floatify(&Default::default())?;
            let floatified_f16 = t.image_floatify(&fp::DType::F16.into())?;
            let floatified_default_f16 = floatified_default.to(&fp::DType::F16.into())?;
            assert!(floatified_default_f16.is_equal(&floatified_f16)?);
        }
        Ok(())
    }
    #[test]
    fn test_image_scale_to_domain() -> StableTorchResult<()> {
        // Float, 6 by 6 pixel of greyscale, top left quadrant set to white.
        let mut d = Tensor::zeros(&[6, 6], &Default::default())?;
        // Top left quadrant +5.0
        d.i_mut((0..3, 0..3))?.fill_f64(5.0)?;
        // Bottom left quadrant at -5.0
        d.i_mut((3..6, 0..3))?.fill_f64(-5.0)?;
        let d = d.image_scale_to_domain()?; // scale it such that we have the full domain.
        assert_eq!(d.f32_ref(&[0, 0])?, &1.0);
        assert_eq!(d.f32_ref(&[5, 0])?, &0.0);
        assert_eq!(d.f32_ref(&[5, 4])?, &0.5);

        Ok(())
    }
    #[test]
    fn test_image_flat_buffer() -> StableTorchResult<()> {
        // Test an rgba image.
        let mut d = Tensor::zeros(&[3, 6, 6], &Default::default())?;
        // Top left, R
        d.i_mut((0, 0..3, 0..3))?.fill_f64(1.0)?;
        // Bottom left, G
        d.i_mut((1, 3..6, 0..3))?.fill_f64(1.0)?;
        // Top right Blue
        d.i_mut((2, 0..3, 3..6))?.fill_f64(1.0)?;
        // Bottom right, white, this also sets the full opacity.
        d.i_mut((.., 3..6, 3..6))?.fill_f64(1.0)?;
        d.save_image("/tmp/fp_rgb_f32_flat.png")?;
        let img = image::ImageReader::open(&"/tmp/fp_rgb_f32_flat.png")?
            .decode()?
            .to_rgb8();
        let flat = img.as_flat_samples();
        println!("{:?}", flat);

        let as_ten = flat.as_ten()?;
        println!("ten: {as_ten:?}");
        println!("{:?}", as_ten.shape());

        Ok(())
    }
}
