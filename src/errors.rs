use super::thiserror::Error;

#[derive(Debug)]
pub enum YakfError {
    // #[error("Concrete type faild to convert from Duration(in f64)")]
    DurationConvertErr,

    // #[error("Cholesky decomposition failed to proceed")]
    CholeskyErr,

    // #[error("Try matrix inverse failed to proceed")]
    InverseErr,
}
