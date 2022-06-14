#![no_std]
#[macro_use]
extern crate log;
extern crate generic_array;
extern crate hifitime;
extern crate nalgebra as na;
// extern crate prost_derive;
extern crate alloc;
extern crate itertools;

extern crate thiserror;

/// Re-export of hifitime
pub mod time {
    pub use hifitime::*;
}

/// Re-export nalgebra
pub mod linalg {
    pub use na::base::*;
    pub use na::RealField;
}

mod errors;
mod filters;
mod tests;