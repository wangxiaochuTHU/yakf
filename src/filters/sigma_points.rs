use crate::errors::YakfError;
use crate::linalg::allocator::Allocator;
use crate::linalg::{DefaultAllocator, DimName, OMatrix, OVector};

/// Any sampling method that implements `SamplingMethod<T, T2>` trait
/// can be used by UKF struct.
pub trait SamplingMethod<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator:
        Allocator<f64, T2> + Allocator<f64, T> + Allocator<f64, T, T2> + Allocator<f64, T, T>,
{
    /// weights for covariance
    fn weights_c(&self) -> &OVector<f64, T2>;

    /// weights for state
    fn weights_m(&self) -> &OVector<f64, T2>;

    /// fixed bases for Minimal skew simplex sampling method
    fn bases(&self) -> Option<&OMatrix<f64, T, T2>>;

    /// check if a sampling method has bases
    fn has_bases(&self) -> bool;

    /// Symmetrically Distributed Sampling Method needs to use `λ + N` scalar, so store it.
    fn get_lamda_plus_n(&self) -> Option<f64>;

    /// sample the states and obtain a serial of samples, form them into a matrix
    fn sampling_states(
        &self,
        p: &OMatrix<f64, T, T>,
        state: &OVector<f64, T>,
    ) -> Result<OMatrix<f64, T, T2>, YakfError>;
}

/// Minimal skew simplex sampling method (MSSS)
/// `T`:  dimension of state.
///
/// `T2`: number of sigma points.
///
/// Minimal skew simplex sampling `n+2` sigma points, i.e. `T2.dim()` = `T::dim() + 2`.
///
/// See the following paper for more detail.
///
/// @inproceedings{julier2002reduced,
///  title={Reduced sigma point filters for the propagation of means and covariances through nonlinear transformations},
///  author={Julier, Simon J and Uhlmann, Jeffrey K},
///  booktitle={Proceedings of the 2002 American Control Conference (IEEE Cat. No. CH37301)},
///  volume={2},
///  pages={887--892},
///  year={2002},
///}
///
#[derive(Debug)]
pub struct MinimalSkewSimplexSampling<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator:
        Allocator<f64, T2> + Allocator<f64, T> + Allocator<f64, T, T2> + Allocator<f64, T, T>,
{
    pub weights: OVector<f64, T2>,
    pub u_bases: Option<OMatrix<f64, T, T2>>,
}

impl<T, T2> SamplingMethod<T, T2> for MinimalSkewSimplexSampling<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator:
        Allocator<f64, T2> + Allocator<f64, T> + Allocator<f64, T, T2> + Allocator<f64, T, T>,
{
    fn weights_c(&self) -> &OVector<f64, T2> {
        &self.weights
    }
    fn weights_m(&self) -> &OVector<f64, T2> {
        &self.weights
    }
    fn bases(&self) -> Option<&OMatrix<f64, T, T2>> {
        self.u_bases.as_ref()
    }
    fn has_bases(&self) -> bool {
        true
    }
    fn get_lamda_plus_n(&self) -> Option<f64> {
        None
    }
    fn sampling_states(
        &self,
        p: &OMatrix<f64, T, T>,
        state: &OVector<f64, T>,
    ) -> Result<OMatrix<f64, T, T2>, YakfError> {
        match p.clone_owned().cholesky() {
            Some(cholesky) => {
                let cho = cholesky.unpack();
                let mut samples: OMatrix<f64, T, T2> = OMatrix::<f64, T, T2>::zeros();
                let bases = self.bases().unwrap();
                for (i, mut col) in samples.column_iter_mut().enumerate() {
                    let u_i = bases.column(i);
                    let chi = state + &cho * u_i;
                    col.copy_from(&chi);
                }

                Ok(samples)
            }
            None => Err(YakfError::CholeskyErr),
        }
    }
}

impl<T, T2> MinimalSkewSimplexSampling<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator:
        Allocator<f64, T2> + Allocator<f64, T, T2> + Allocator<f64, T> + Allocator<f64, T, T>,
{
    #[allow(dead_code)]
    pub fn build(w0: f64) -> Result<Self, YakfError> {
        let mut sampling = Self::empty()?;
        sampling.set_weights(w0);
        sampling.expand_bases(T::dim());
        Ok(sampling)
    }

    #[allow(dead_code)]
    /// generate an object with fields filled with zeros.
    fn empty() -> Result<Self, YakfError> {
        if T2::dim() != T::dim() + 2 {
            error!("Weights dimention should be set as (n + 2), where n is the state dimention.");
            Err(YakfError::DimensionMismatchErr)
        } else {
            Ok(Self {
                weights: OVector::<f64, T2>::zeros(),
                u_bases: None,
            })
        }
    }

    /// sets the parameter that is required by MSSS
    fn set_weights(&mut self, w0: f64) {
        // n stands for state dimention
        let n = T::dim();

        // weight 0 is specified.
        self.weights[0] = w0;

        // weights [1, 2]
        for i in 1..3 {
            self.weights[i] = (1.0 - self.weights[0]) / 2_f64.powi(n as i32);
        }

        // weights [3, n+1]
        for i in 3..n + 2 {
            self.weights[i] = 2_f64.powi(i as i32 - 2) * self.weights[1];
        }
    }

    #[allow(dead_code)]
    /// positive scaling parameter `a` ranges from [1-e4, 1]
    /// TODO: Add scaling weights.
    fn scale_weights(&mut self, _a: f64) {}

    /// Expand vector sequence w.r.t weights.
    fn expand_bases(&mut self, n: usize) {
        // Expands the bases in a cumbersome way.
        // TODO: How to optimize this progress?

        // cols to save columns
        let mut cols: OMatrix<f64, T, T2> = OMatrix::<f64, T, T2>::zeros();

        // col-0 is zero column
        let col0 = OVector::<f64, T>::zeros();

        let mut w_iter =
            self.weights
                .iter()
                .enumerate()
                .map(|(i, w)| if i == 0 { 0.0 } else { 1.0 / (2.0 * w).sqrt() });

        // col-1 can be expanded as
        let mut col1 = OVector::<f64, T>::zeros();
        w_iter.next();
        for k in 0..n {
            col1[k] = -w_iter.next().unwrap(); // this is because that we want: col1[k] = w_iter[k+1]
        }

        for i in 0..T2::dim() {
            if i == 0 {
                // save col0
                cols.set_column(i, &col0);
            } else if i == 1 {
                // save col1
                cols.set_column(i, &col1);
            } else {
                // iterates for yielding other cols in [2, n+1] and save them
                let rev_idx = i - 2;
                let mut new_col = cols.column(i - 1).clone_owned();
                for k in 0..rev_idx {
                    new_col[k] = 0.0;
                }
                new_col[rev_idx] = -new_col[rev_idx];
                cols.set_column(i, &new_col);
            }
        }

        // Now, columns should be (n+2) in count. Finally, build u_bases from these columns.
        self.u_bases = Some(OMatrix::<f64, T, T2>::from(cols));
    }
}

/* Symmetrically-Distributed Sampling Method */

/// Symmetrically-Distributed Sampling Method (SDS)
/// `T`:  dimension of state.
///
/// `T2`: number of sigma points.
///
/// Symmetrically-distributed sampling `2n+1` sigma points, i.e. `T2.dim()` = `2 * T::dim() + 1`.
///
/// See the following paper for more detail.
///
///@inproceedings{wan2000unscented,
/// title={The unscented Kalman filter for nonlinear estimation},
/// author={Wan, Eric A and Van Der Merwe, Rudolph},
/// booktitle={Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (Cat. No. 00EX373)},
/// pages={153--158},
/// year={2000},
/// organization={Ieee}
///}
///

#[derive(Debug)]
pub struct SymmetricallyDistributedSampling<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator:
        Allocator<f64, T2> + Allocator<f64, T> + Allocator<f64, T, T2> + Allocator<f64, T, T>,
{
    pub weights_c: OVector<f64, T2>,
    pub weights_m: OVector<f64, T2>,
    pub u_bases: OMatrix<f64, T, T2>,
    pub k: f64,
}

impl<T, T2> SamplingMethod<T, T2> for SymmetricallyDistributedSampling<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator:
        Allocator<f64, T2> + Allocator<f64, T> + Allocator<f64, T, T2> + Allocator<f64, T, T>,
{
    fn weights_c(&self) -> &OVector<f64, T2> {
        &self.weights_c
    }
    fn weights_m(&self) -> &OVector<f64, T2> {
        &self.weights_m
    }
    fn bases(&self) -> Option<&OMatrix<f64, T, T2>> {
        None
    }
    fn has_bases(&self) -> bool {
        false
    }
    fn get_lamda_plus_n(&self) -> Option<f64> {
        Some(self.k)
    }

    fn sampling_states(
        &self,
        p: &OMatrix<f64, T, T>,
        state: &OVector<f64, T>,
    ) -> Result<OMatrix<f64, T, T2>, YakfError> {
        match p.clone().cholesky() {
            Some(cholesky) => {
                let cho = cholesky.unpack();
                let mut samples: OMatrix<f64, T, T2> = OMatrix::<f64, T, T2>::zeros();
                let sqrt_lamda_plus_n = self.get_lamda_plus_n().unwrap().sqrt();
                for (i, mut col) in samples.column_iter_mut().enumerate() {
                    if i == 0 {
                        let chi = state;
                        col.copy_from(&chi);
                    } else if i <= T::dim() {
                        let chi = state + sqrt_lamda_plus_n * &cho.column(i);
                        col.copy_from(&chi);
                    } else {
                        let chi = state - sqrt_lamda_plus_n * &cho.column(i);
                        col.copy_from(&chi);
                    };
                }

                Ok(samples)
            }
            None => Err(YakfError::CholeskyErr),
        }
    }
}

impl<T, T2> SymmetricallyDistributedSampling<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator:
        Allocator<f64, T2> + Allocator<f64, T, T2> + Allocator<f64, T> + Allocator<f64, T, T>,
{
    #[allow(dead_code)]
    /// a, stands for the spread extent from the mean. normally ranged in [1e-4, 1] and, typically, `a = 1e-3`.
    ///
    /// b, stands for the emphasis put on the 0-th sample. normally, b = `None` means using `b = Some(2.0)` by default, which is optimal for Gaussian distribution.
    ///
    /// k, stands for a third parameter. normally, `k = None` means using `k = Some(0.0)`, which is the most common case.
    pub fn build(a: f64, b: Option<f64>, k: Option<f64>) -> Result<Self, YakfError> {
        // by default b = 2.0, unless specified.
        let b = match b {
            Some(v) => v,
            None => 2_f64,
        };

        // by default k = 0.0, unless specified.
        let k = match k {
            Some(v) => v,
            None => 0_f64,
        };

        let mut sampling = Self::empty()?;
        sampling.set_weights(a, b, k);
        Ok(sampling)
    }

    #[allow(dead_code)]
    /// generate an object with fields filled with zeros.
    fn empty() -> Result<Self, YakfError> {
        if T2::dim() != 2 * T::dim() + 1 {
            error!("Weights dimention should be set as (n + 2), where n is the state dimention.");
            Err(YakfError::DimensionMismatchErr)
        } else {
            Ok(Self {
                weights_c: OVector::<f64, T2>::zeros(),
                weights_m: OVector::<f64, T2>::zeros(),
                u_bases: OMatrix::<f64, T, T2>::zeros(),
                k: 0_f64,
            })
        }
    }
    /// sets the parameter that is required by MSSS
    fn set_weights(&mut self, a: f64, b: f64, k: f64) -> Option<f64> {
        // n stands for state dimention
        let n = T::dim();

        let lamda = (a as f64).powi(2) * (n as f64 + k) - n as f64;
        let lamda_plus_n = lamda + n as f64;

        // weight 0 is specified.
        self.weights_m[0] = lamda / lamda_plus_n;
        self.weights_c[0] = self.weights_m[0] + (1.0 - (a as f64).powi(2) + b);
        for i in 1..2 * n + 1 {
            self.weights_m[i] = 0.5 / lamda_plus_n;
            self.weights_c[i] = self.weights_c[i];
        }
        self.k = lamda_plus_n;
        Some(lamda_plus_n)
    }
}
