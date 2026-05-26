//! Super thin helper function for f16 floats
//!
//! Mostly just to be able to print them.

//https://github.com/iwanders/cbor/blob/8273ced6972ece28b09cc726058b0f8586e9b544/test/test_shortfloat.cpp#L42

#[derive(Copy, Clone)]
pub(crate) struct F16(u16);

impl F16 {
    pub fn from_u16(v: u16) -> Self {
        Self(v)
    }
    pub fn into_f64(self) -> f64 {
        fn ldexp(x: f64, exp: i32) -> f64 {
            x * (2.0f64).powi(exp)
        }
        let h = self.0;

        let exp = ((h >> 10) & 0x1f) as i32;
        let mantissa = (h & 0x3ff) as u32;
        let val: f64;
        if exp == 0 {
            val = ldexp(mantissa as f64, -24);
        } else if exp != 31 {
            val = ldexp(mantissa as f64 + 1024.0, exp - 25);
        } else {
            val = if mantissa == 0 {
                f64::INFINITY
            } else {
                f64::NAN
            };
        }
        if h & 0x8000 != 0 { -val } else { val }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_flash_powder_f16_conversion() {
        assert_eq!(F16(0x0000).into_f64(), 0.0);
        assert_eq!(F16(0x8000).into_f64(), -0.0);
        assert_eq!(F16(0x3c00).into_f64(), 1.0);
        assert_eq!(F16(0xc000).into_f64(), -2.0);
        assert_eq!(F16(0x7c00).into_f64(), f64::INFINITY);
        assert_eq!(F16(0xfC00).into_f64(), f64::NEG_INFINITY);
        assert_eq!(F16(0x7e00).into_f64().is_nan(), true);
        assert_eq!(F16(0x4268).into_f64(), 3.203125);
        assert_eq!(F16(0x78EA).into_f64(), 40256.0);
    }
}
