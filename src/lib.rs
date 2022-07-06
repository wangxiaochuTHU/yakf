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
    pub use super::dfilters::{dekf::*, dsigma_points::*, dstate::*, dukf::*};
    pub use super::errors::YakfError;
    pub use super::filters::{sigma_points::*, state::*, ukf::*};
    pub use super::lie;
    pub use super::lie::se3::{self, Alg6, Grp6, One2OneMapSE, Vec6};
    pub use super::lie::so3::{self, sosekf::SOEKF, Alg3, Grp3, One2OneMap, Vec3};
}

pub mod dfilters;
pub mod errors;
pub mod filters;
pub mod lie;
// pub mod sfilters;
mod tests;
