use crate::linalg::allocator::Allocator;
use crate::linalg::{DefaultAllocator, DimName, OMatrix, OVector};

/// Any sampling method that implements `SamplingMethod<T, T2>` trait
/// can be used by UKF struct.
pub trait SamplingMethod<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator: Allocator<f64, T2> + Allocator<f64, T> + Allocator<f64, T, T2>,
{
    fn weights_c(&self) -> &OVector<f64, T2>;
    fn weights_m(&self) -> &OVector<f64, T2>;
    fn bases(&self) -> &OMatrix<f64, T, T2>;
}

/// Minimal skew simplex sampling method (MSSS)
///
/// `T`:  dimension of state.
///
/// `T2`: number of sigma points.
///
/// Minimal skew simplex sampling `n+2` sigma points, i.e. `T2.dim()` = `T::dim() + 2`.
#[derive(Debug)]
pub struct MinimalSkewSimplexSampling<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator: Allocator<f64, T2> + Allocator<f64, T> + Allocator<f64, T, T2>,
{
    pub weights: OVector<f64, T2>,
    pub u_bases: OMatrix<f64, T, T2>,
}

impl<T, T2> SamplingMethod<T, T2> for MinimalSkewSimplexSampling<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator: Allocator<f64, T2> + Allocator<f64, T> + Allocator<f64, T, T2>,
{
    fn weights_c(&self) -> &OVector<f64, T2> {
        &self.weights
    }
    fn weights_m(&self) -> &OVector<f64, T2> {
        &self.weights
    }
    fn bases(&self) -> &OMatrix<f64, T, T2> {
        &self.u_bases
    }
}

impl<T, T2> MinimalSkewSimplexSampling<T, T2>
where
    T: DimName,
    T2: DimName,
    DefaultAllocator: Allocator<f64, T2> + Allocator<f64, T, T2> + Allocator<f64, T>,
{
    #[allow(dead_code)]
    pub fn build(w0: f64) -> Self {
        let mut sampling = Self::empty();
        sampling.set_weights(w0);
        sampling.expand_bases(T::dim());
        sampling
    }

    #[allow(dead_code)]
    /// generate an object with fields filled with zeros.
    fn empty() -> Self {
        Self {
            weights: OVector::<f64, T2>::zeros(),
            u_bases: OMatrix::<f64, T, T2>::zeros(),
        }
    }

    /// sets the parameter that is required by MSSS
    fn set_weights(&mut self, w0: f64) {
        let num_weights = self.weights.len();
        if num_weights < 3 {
            error!("Weights dimention should be set as (n + 2), where n is the state dimention.")
        }

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
        self.u_bases.copy_from(&cols);
    }
}
