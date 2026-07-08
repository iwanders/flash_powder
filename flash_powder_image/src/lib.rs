//! Helper tooling for interop with [image].
//!

use flash_powder as fp;
use fp::Tensor;

use anyhow::bail;
use flash_powder::prelude::*;
pub use image;

use fp::StableTorchResult;

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
    /// All integer types are expected to fit within a byte [0, 255]. All (supported) float types in [0.0, 1.0].
    /// All images are exported as [0, 255] u8.
    ///
    /// The pytorch side only accepts (B x C x H x W), with an argument to specify number per row, this function puts
    /// the B dimension always on the same row, but you can do  (V x B x C x H x W), where V is stacking rows.
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
            3 => v.unsqueeze(0)?.into_unsqueeze(0)?,
            4 => v.unsqueeze(0)?,
            5 => v,
            _ => unreachable!(),
        };
        assert_eq!(v.dim(), 5);

        // Perform a grandiose swap to interleave the data.
        //let channels_stacked = t.permute(&[2, 0, 1])?;
        //
        // Next, we move the third from last to the end, moving height and width left.
        let dimension_count = v.dim();
        let current_order: Vec<usize> = (0..dimension_count).into_iter().collect();
        let channel_count = current_order[dimension_count - 3];
        let height = current_order[dimension_count - 2];
        let width = current_order[dimension_count - 1];
        let mut desired_channel_order = current_order;
        desired_channel_order[dimension_count - 3] = height;
        desired_channel_order[dimension_count - 2] = width;
        desired_channel_order[dimension_count - 1] = channel_count;

        // Permute it, and also make sure we get a contiguous block of data back.
        let v = v.permute(&desired_channel_order)?.contiguous()?;

        // Cool now we have an interleaved image, but it may still be varying dimensionality at the front.
        // [V, B, C, H, W]

        let image_width = v.isize(-2);
        let image_height = v.isize(-3);
        let image_per_row = if v.dim() > 3 { v.isize(-4) } else { 1 };
        let image_rows = if v.dim() == 5 { v.isize(-5) } else { 1 };

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
        let v = v.permute(&[0, 2, 1, 3, 4])?.contiguous()?;

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

pub trait TensorFromImage {
    /// Like [decode_image](https://docs.pytorch.org/vision/main/generated/torchvision.io.decode_image.html#torchvision.io.decode_image)
    ///
    /// The values of the output tensor are in uint8 in [0, 255] for most cases.
    ///
    /// output (Tensor[image_channels, image_height, image_width])
    fn from_dynamic_image(dynamic_image: &image::DynamicImage) -> StableTorchResult<Tensor>;

    /// Read an image from disk.
    fn read_image<Q>(path: Q) -> StableTorchResult<Tensor>
    where
        Q: AsRef<std::path::Path>;
}

impl TensorFromImage for Tensor {
    fn from_dynamic_image(img: &image::DynamicImage) -> StableTorchResult<Tensor> {
        let color = img.color();
        let channels = color.channel_count() as usize;
        let bytes_per_pixel = color.bytes_per_pixel() as usize;
        let width = img.width() as usize;
        let height = img.height() as usize;

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
        t.data_mut()?.copy_from_slice(img.as_bytes());

        // And finally, perform the channel swap.
        let channels_stacked = t.permute(&[2, 0, 1])?;

        // ANd return an owned version.
        channels_stacked.to_owned()
    }

    fn read_image<Q>(path: Q) -> StableTorchResult<Tensor>
    where
        Q: AsRef<std::path::Path>,
    {
        let img = image::ImageReader::open(path)?.decode()?;
        Self::from_dynamic_image(&img)
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
        d.save_image("/tmp/fp_rgba_f32.png").unwrap();
        let img = image::ImageReader::open(&"/tmp/fp_rgba_f32.png")?.decode()?;
        assert!(matches!(img, image::DynamicImage::ImageRgba8(_)));
        let v = Tensor::read_image("/tmp/fp_rgba_f32.png")?
            .to(&fp::DType::F32.into())?
            .div(&f32_255)?;
        assert!(d.is_equal(&v)?);
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

        // These batches we can't really test... since we make a composite.
        let mut d = Tensor::zeros(&[3, 3, 6, 6], &Default::default())?;
        d.i_mut((0, 0, 0..6, 0..6))?.fill_f64(1.0)?; // first image in batch red
        d.i_mut((1, 2, 0..6, 0..6))?.fill_f64(1.0)?; // second image in batch blue.
        d.i_mut((2, 1, 0..6, 0..6))?.fill_f64(1.0)?; // third image in batch green.
        d.save_image("/tmp/fp_rgb_b2.png").unwrap();
        let img = image::ImageReader::open(&"/tmp/fp_rgb_b2.png")?.decode()?;
        assert!(matches!(img, image::DynamicImage::ImageRgb8(_)));
        let back = Tensor::from_dynamic_image(&img)?;
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
        let back = Tensor::from_dynamic_image(&img)?;
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
}
