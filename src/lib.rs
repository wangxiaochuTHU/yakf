#![no_std]
#[macro_use]
extern crate log;
extern crate alloc;
extern crate hifitime;
extern crate itertools;
extern crate nalgebra as na;

/// Re-export of hifitime
pub mod time {
    pub use hifitime::*;
}

/// Re-export nalgebra
pub mod linalg {
    pub use na::base::*;
    pub use na::RealField;
}

/// Export yakf
pub mod kf {
    pub use super::errors::YakfError;
    pub use super::filters::sigma_points::*;
    pub use super::filters::state::*;
    pub use super::filters::ukf::*;
}

mod errors;
mod filters;
mod tests;
