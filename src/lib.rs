//! Sliding discrete fourier transform library for real input, optimized for
//! no-std environment.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

use core::f32::consts::PI;

use num_complex::{Complex, Complex32};

/// Sliding DFT implementation.
#[derive(Debug, Clone, Copy)]
pub struct SlidingDft<ValuesBuffer, TransformedBuffer, CoefficientsStorage> {
    buffer: ValuesBuffer,
    transformed: TransformedBuffer,
    rotation_coefficients: CoefficientsStorage,
    begin_ptr: usize,
}

/// Sliding DFT construction error.
///
/// When the `std` feature is activated, this type also implements the
/// [Error][std::error::Error] trait.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    /// Length of the image buffer does not match length of the coefficients buffer.
    ImageCoefficientsMismatch {
        /// Length of the image buffer.
        image_length: usize,

        /// Length of the coefficients buffer.
        coefficients_length: usize,
    },
    /// The coefficients buffer is too big.
    TooMuchCoefficients {
        /// Length of the coefficients buffer.
        coefficients_length: usize,

        /// Maximum expected length of the coefficients buffer.
        maximum_expected: usize,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::ImageCoefficientsMismatch {
                image_length,
                coefficients_length,
            } => {
                write!(
                    f,
                    "Length of the image buffer ({}) does not \
                    match length of the coefficients buffer ({})",
                    image_length, coefficients_length
                )
            }
            Error::TooMuchCoefficients {
                coefficients_length,
                maximum_expected,
            } => {
                write!(
                    f,
                    "The coefficients buffer is too big ({} > {})",
                    coefficients_length, maximum_expected
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl<ValuesBuffer, TransformedBuffer, CoefficientsStorage>
    SlidingDft<ValuesBuffer, TransformedBuffer, CoefficientsStorage>
where
    ValuesBuffer: AsRef<[f32]> + AsMut<[f32]>,
    TransformedBuffer: AsRef<[Complex32]> + AsMut<[Complex32]>,
    CoefficientsStorage: AsRef<[Complex32]> + AsMut<[Complex32]>,
{
    /// Initializes the algorithm with the given buffers.
    ///
    /// It is advised to either manually zeroize the provided `values_buffer`
    /// and `transformed_storage` buffers, or call
    /// [zeroize][SlidingDft::zeroize] after the initialization. Anyhow,
    /// `transformed_storage` should contain the fourier transform of the
    /// `values_buffer`, otherwise the algorithm might produce unexpected
    /// results (however, no undefined behavior is involved).
    ///
    /// # Arguments
    /// * `values_buffer`: the buffer is used as a circular buffer to store the
    ///   provided by [SlidingDft::add_point] values. Length of the buffer
    ///   represents the amount of samples stored at any given time.
    /// * `transformed_storage`: the buffer will contain sliding fourier
    ///   transform of the input values. Since the algorithm works only with
    ///   real inputs, it only makes sense to store the N/2 amount of values (or
    ///   less), where N is length of the `values_buffer`.
    /// * `coefficients_storage`: the buffer is initialized by this call with
    ///   complex coefficients used when advancing the algorithm. The length of
    ///   the buffer must be the same as of `transformed_storage`.
    ///
    /// # Errors
    /// Error is returned when:
    /// * Length of the `image` does not match length of the
    ///   `coefficients_storage`.
    /// * Length of the `coefficients_storage` is greater than half of length of
    ///   the `buffer`.
    pub fn new(
        values_buffer: ValuesBuffer,
        transformed_storage: TransformedBuffer,
        mut coefficients_storage: CoefficientsStorage,
    ) -> Result<Self, Error> {
        let coefficients_length = coefficients_storage.as_ref().len();
        let image_length = transformed_storage.as_ref().len();
        let buffer_length = values_buffer.as_ref().len();

        if coefficients_length != image_length {
            return Err(Error::ImageCoefficientsMismatch {
                image_length,
                coefficients_length,
            });
        } else if coefficients_length > buffer_length / 2 {
            return Err(Error::TooMuchCoefficients {
                coefficients_length,
                maximum_expected: buffer_length / 2,
            });
        }

        assert_eq!(
            coefficients_storage.as_ref().len(),
            transformed_storage.as_ref().len()
        );
        assert!(coefficients_storage.as_ref().len() <= values_buffer.as_ref().len() / 2);

        let buffer_length_f32 = buffer_length as f32;
        coefficients_storage
            .as_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(omega, value)| {
                *value = Complex::new(0., 2. * PI * (omega as f32) / buffer_length_f32).exp();
            });
        Ok(Self {
            buffer: values_buffer,
            transformed: transformed_storage,
            begin_ptr: 0,
            rotation_coefficients: coefficients_storage,
        })
    }

    /// Zeroizes the internal buffers.
    pub fn zeroize(&mut self) {
        self.buffer.as_mut().iter_mut().for_each(|v| *v = 0.);
        self.transformed
            .as_mut()
            .iter_mut()
            .for_each(|v| *v = Complex32::new(0., 0.));
    }

    /// Adds a point to the DFT.
    pub fn add_point(&mut self, value: f32) {
        // Safety: we are sure that `self.begin_ptr` is always in range.
        let delta = value - unsafe { *self.buffer.as_ref().get_unchecked(self.begin_ptr) };

        self.transformed
            .as_mut()
            .iter_mut()
            .zip(self.rotation_coefficients.as_ref())
            .for_each(|(image, rotation)| {
                *image = (*image + delta) * rotation;
            });

        let buffer_length = self.buffer.as_ref().len();

        // Safety: we are sure that `(self.begin_ptr + self.length - 1) %
        // self.length` is always in range.
        unsafe {
            *self
                .buffer
                .as_mut()
                .get_unchecked_mut((self.begin_ptr + buffer_length - 1) % buffer_length) = value
        };

        // That's how we guarantee safety of the above blocks.
        self.begin_ptr = (self.begin_ptr + 1) % buffer_length;
    }

    /// Returns a reference to the current transformed values.
    pub fn transformed(&self) -> &[Complex32] {
        self.transformed.as_ref()
    }

    /// Releases the underlying buffers.
    pub fn release_buffers(self) -> (ValuesBuffer, TransformedBuffer, CoefficientsStorage) {
        (self.buffer, self.transformed, self.rotation_coefficients)
    }
}

#[cfg(test)]
mod tests {
    use crate::SlidingDft;
    use core::f32::consts::PI;
    use num_complex::Complex32;

    macro_rules! assert_almost_eq {
        ($left:expr, $right:expr, $precision:expr ) => {
            let difference = ($left - $right).abs();
            if difference > $precision {
                panic!("|{} - {}| = {} > {}", $left, $right, difference, $precision);
            }
        };
        ($left:expr, $right:expr, $precision:expr, $message:literal $(,$args:expr)* ) => {
            let difference = ($left - $right).abs();
            if difference > $precision {
                panic!(concat!("|{} - {}| = {} > {}; ", $message), $left, $right, difference, $precision $(,$args)*);
            }
        };
    }

    #[test]
    fn sliding_dft_one_wave() {
        // We are using 2048 points for the sake of the speed.
        const POINTS: usize = 2048;
        const FREQUENCIES: usize = POINTS / 2;
        const FREQUENCIES_F32: f32 = POINTS as f32;

        let mut values_buffer = [0f32; POINTS];
        let mut image_storage = [Complex32::new(0., 0.); FREQUENCIES];
        let mut coefficients_storage = [Complex32::new(0., 0.); FREQUENCIES];

        let mut sliding_dft = SlidingDft::new(
            &mut values_buffer[..],
            &mut image_storage[..],
            &mut coefficients_storage[..],
        )
        .unwrap();

        for frequency in 1..FREQUENCIES {
            let frequency_f32 = frequency as f32;

            sliding_dft.zeroize();

            for id in 0..POINTS {
                let value = (2. * PI * frequency_f32 * (id as f32) / FREQUENCIES_F32).sin();
                sliding_dft.add_point(value);
            }

            for (freq, value) in sliding_dft
                .transformed()
                .iter()
                .take(FREQUENCIES)
                .enumerate()
            {
                let amplitude = value.norm() / FREQUENCIES_F32 * 2.;
                if freq == frequency {
                    assert_almost_eq!(amplitude, 1.0, 1E-4, "base frequency = {}", frequency);
                } else {
                    assert_almost_eq!(amplitude, 0.0, 1E-3, "base frequency = {}", frequency);
                }
            }
        }
    }

    #[test]
    fn sliding_dft_two_waves() {
        const POINTS: usize = 4096;

        // Let's test only half of the available frequencies for the sake of speed.
        const FREQUENCIES: usize = POINTS / 4;
        const FREQUENCIES_F32: f32 = POINTS as f32;

        let mut values_buffer = [0f32; POINTS];
        let mut image_storage = [Complex32::new(0., 0.); FREQUENCIES];
        let mut coefficients_storage = [Complex32::new(0., 0.); FREQUENCIES];

        let mut sliding_dft = SlidingDft::new(
            &mut values_buffer[..],
            &mut image_storage[..],
            &mut coefficients_storage[..],
        )
        .unwrap();

        let second_frequency = 100;
        let second_frequency_f32 = second_frequency as f32;

        for first_frequency in 1..FREQUENCIES {
            let frequency_f32 = first_frequency as f32;

            sliding_dft.zeroize();

            for id in 0..POINTS {
                let wave_1 = (2. * PI * frequency_f32 * (id as f32) / FREQUENCIES_F32).sin();
                let wave_2 = (2. * PI * second_frequency_f32 * (id as f32) / FREQUENCIES_F32).sin();
                sliding_dft.add_point(wave_1 + wave_2);
            }

            for (freq, value) in sliding_dft
                .transformed()
                .iter()
                .take(FREQUENCIES)
                .enumerate()
            {
                let amplitude = value.norm() / FREQUENCIES_F32 * 2.;
                if freq == first_frequency && freq == second_frequency {
                    // The resonance frequency :)
                    assert_almost_eq!(amplitude, 2.0, 1E-4, "base frequency = {}", first_frequency);
                } else if freq == first_frequency || freq == second_frequency {
                    assert_almost_eq!(amplitude, 1.0, 1E-4, "base frequency = {}", first_frequency);
                } else {
                    assert_almost_eq!(amplitude, 0.0, 1E-4, "base frequency = {}", first_frequency);
                }
            }
        }
    }
}
