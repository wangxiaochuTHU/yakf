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
    /// Lie Theory Related
    pub use super::lie;
    /// Lie Theory SE(2)
    pub use super::lie::se2::{self, Alg3, Grp3, One2OneMapSE2, Vec3};
    /// Lie Theory SE(3)
    pub use super::lie::se3::{self, Alg6, Grp6, One2OneMapSE3, Vec6};
    /// Lie Theory SO(2)
    pub use super::lie::so2::{self, Alg1, Grp1, One2OneMapSO2, Vec1};
    /// Lie Theory SO(3)
    pub use super::lie::so3::{self, One2OneMapSO3};
}

pub mod dfilters;
pub mod errors;
pub mod filters;
pub mod lie;
// pub mod sfilters;
mod tests;
