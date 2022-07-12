use super::dfixed::{dmatrix_zeros, dvector_zeros};
use crate::errors::YakfError;
use crate::linalg::{DMatrix, DVector};
use num_traits::float::FloatCore;
/// Any sampling method that implements `DSamplingMethod` trait
/// can be used by DUKF struct.
pub trait DSamplingMethod {
    fn weights_c(&self) -> &DVector<f64>;
    fn weights_m(&self) -> &DVector<f64>;
    fn bases(&self) -> Option<&DMatrix<f64>>;
    fn has_bases(&self) -> bool;
    fn get_k(&self) -> Option<f64>;
    fn sampling_states(
        &self,
        p: &DMatrix<f64>,
        state: &DVector<f64>,
    ) -> Result<DMatrix<f64>, YakfError>;
}

/// Minimal skew simplex sampling method (MSSS) for DUKF
///
#[derive(Debug)]
pub struct DMinimalSkewSimplexSampling {
    pub weights: DVector<f64>,
    pub u_bases: Option<DMatrix<f64>>,
}

impl DSamplingMethod for DMinimalSkewSimplexSampling {
    fn weights_c(&self) -> &DVector<f64> {
        &self.weights
    }
    fn weights_m(&self) -> &DVector<f64> {
        &self.weights
    }
    fn bases(&self) -> Option<&DMatrix<f64>> {
        self.u_bases.as_ref()
    }
    fn has_bases(&self) -> bool {
        true
    }
    fn get_k(&self) -> Option<f64> {
        None
    }
    fn sampling_states(
        &self,
        p: &DMatrix<f64>,
        state: &DVector<f64>,
    ) -> Result<DMatrix<f64>, YakfError> {
        match p.clone_owned().cholesky() {
            Some(cholesky) => {
                let cho = cholesky.unpack();
                let nrows = state.len();
                let ncols = nrows + 2;
                let mut samples = dmatrix_zeros(nrows, ncols);
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

impl DMinimalSkewSimplexSampling {
    #[allow(dead_code)]
    pub fn build(w0: f64, n: usize) -> Result<Self, YakfError> {
        let mut sampling = Self::empty(n)?;
        // info!("sampling = {:?}", sampling);
        // panic!();
        sampling.set_weights(w0, n);
        sampling.expand_bases(n);
        Ok(sampling)
    }

    #[allow(dead_code)]
    /// generate an object with fields filled with zeros.
    fn empty(n: usize) -> Result<Self, YakfError> {
        Ok(Self {
            weights: dvector_zeros(n + 2),
            u_bases: None,
        })
    }

    /// sets the parameter that is required by MSSS
    fn set_weights(&mut self, w0: f64, n: usize) {
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
        let mut cols = dmatrix_zeros(n, n + 2);

        // col-0 is zero column
        let col0 = dvector_zeros(n);

        let mut w_iter =
            self.weights
                .iter()
                .enumerate()
                .map(|(i, w)| if i == 0 { 0.0 } else { 1.0 / libm::sqrt(2.0 * w) });

        // col-1 can be expanded as
        let mut col1 = dvector_zeros(n);
        w_iter.next();
        for k in 0..n {
            col1[k] = -w_iter.next().unwrap(); // this is because that we want: col1[k] = w_iter[k+1]
        }

        for i in 0..n + 2 {
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
        self.u_bases = Some(DMatrix::<f64>::from(cols));
    }
}

#[derive(Debug)]
pub struct DSymmetricallyDistributedSampling {
    pub n: usize,
    pub weights_c: DVector<f64>,
    pub weights_m: DVector<f64>,
    pub u_bases: DMatrix<f64>,
    pub k: f64,
}

impl DSamplingMethod for DSymmetricallyDistributedSampling {
    fn weights_c(&self) -> &DVector<f64> {
        &self.weights_c
    }
    fn weights_m(&self) -> &DVector<f64> {
        &self.weights_m
    }
    fn bases(&self) -> Option<&DMatrix<f64>> {
        None
    }
    fn has_bases(&self) -> bool {
        false
    }
    fn get_k(&self) -> Option<f64> {
        Some(self.k)
    }

    fn sampling_states(
        &self,
        p: &DMatrix<f64>,
        state: &DVector<f64>,
    ) -> Result<DMatrix<f64>, YakfError> {
        match p.clone().cholesky() {
            Some(cholesky) => {
                let cho = cholesky.unpack();
                let mut samples: DMatrix<f64> = dmatrix_zeros(self.n, 2 * self.n + 1);
                let sqrt_lamda_plus_n =  libm::sqrt(self.get_k().unwrap());
                for (i, mut col) in samples.column_iter_mut().enumerate() {
                    if i == 0 {
                        let chi = state;
                        col.copy_from(&chi);
                    } else if i <= self.n {
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

impl DSymmetricallyDistributedSampling {
    #[allow(dead_code)]
    /// a, stands for the spread extent from the mean. normally ranged in [1e-4, 1] and, typically, `a = 1e-3`.
    ///
    /// b, stands for the emphasis put on the 0-th sample. normally, b = `None` means using `b = Some(2.0)` by default, which is optimal for Gaussian distribution.
    ///
    /// k, stands for a third parameter. normally, `k = None` means using `k = Some(0.0)`, which is the most common case.
    pub fn build(a: f64, b: Option<f64>, k: Option<f64>, n: usize) -> Result<Self, YakfError> {
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

        let mut sampling = Self::empty(n)?;
        sampling.set_weights(a, b, k);
        Ok(sampling)
    }

    #[allow(dead_code)]
    /// generate an object with fields filled with zeros.
    fn empty(n: usize) -> Result<Self, YakfError> {
        Ok(Self {
            n: n,
            weights_c: dvector_zeros(2 * n + 1),
            weights_m: dvector_zeros(2 * n + 1),
            u_bases: dmatrix_zeros(n, 2 * n + 1),
            k: 0_f64,
        })
    }
    /// sets the parameter that is required by MSSS
    fn set_weights(&mut self, a: f64, b: f64, k: f64) -> Option<f64> {
        // n stands for state dimention
        let n = self.n;

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
