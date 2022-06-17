use crate::alloc::vec;

use crate::linalg::{DMatrix, DVector};
/// TODO: I cann't figure out why `DMatrix::<f64>::zeros(nrows,ncols)` doesn't work, so init a matrix in this way.
pub fn dmatrix_zeros(nrows: usize, ncols: usize) -> DMatrix<f64> {
    let empty_col = DVector::<f64>::from_row_slice(vec![0_f64; nrows].as_slice());
    DMatrix::<f64>::from_columns(vec![empty_col; ncols].as_slice())
}

/// TODO: I cann't figure out why `DVector::<f64>::zeros(nrows)` doesn't work, so init a vector in this way.
pub fn dvector_zeros(nrows: usize) -> DVector<f64> {
    DVector::<f64>::from_row_slice(vec![0_f64; nrows].as_slice())
}
