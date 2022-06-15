#[derive(Debug)]
pub enum YakfError {
    #[allow(dead_code)]
    // #[error("Concrete type faild to convert from Duration(in f64)")]
    DurationConvertErr,

    // #[error("Cholesky decomposition failed to proceed")]
    CholeskyErr,

    // #[error("Try matrix inverse failed to proceed")]
    InverseErr,

    // #[error("The dimensions are incorrectly set")]
    DimensionMismatchErr,
}
