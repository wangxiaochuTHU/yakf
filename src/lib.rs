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
    pub use super::dfilters::{dsigma_points::*, dstate::*, dukf::*};
    pub use super::errors::YakfError;
    pub use super::filters::{sigma_points::*, state::*, ukf::*};
}

pub mod dfilters;
mod errors;
pub mod filters;
mod tests;
