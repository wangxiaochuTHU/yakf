use super::so3::{
    exp as exp3, hat as hat3, log as log3, vee as vee3, Alg3, Grp3, Vec3, SMALL_FLOAT, SO3,
};
/// Generators of SE(3) are used as below:
///
/// E1 =    |0  0   0   1|
///         |0  0   0   0|
///         |0  0   0   0|
///         |0  0   0   0|
///
/// E2 =    |0  0   0   0|
///         |0  0   0   1|
///         |0  0   0   0|
///         |0  0   0   0|
///
/// E3 =    |0  0   0   0|
///         |0  0   0   0|
///         |0  0   0   1|
///         |0  0   0   0|
///
/// E4 =    |0  0   0   0|
///         |0  0   -1  0|
///         |0  1   0   0|
///         |0  0   0   0|
///
/// E5 =    |0  0   1   0|
///         |0  0   0   0|
///         |-1 0   0   0|
///         |0  0   0   0|
///
/// E6 =    |0  -1  0   0|
///         |1  0   0   0|
///         |0  0   0   0|
///         |0  0   0   0|
///  se(3) vector follows the order: [ ρ , θ ]
use nalgebra::ComplexField;

use crate::alloc::borrow::ToOwned;
use crate::alloc::vec::Vec;
use crate::time::{Duration, Epoch};
use core::convert::AsRef;
use core::convert::From;
use core::f64::consts::PI;

use crate::errors::YakfError;
use crate::linalg::allocator::Allocator;
use crate::linalg::{DefaultAllocator, DimName, OMatrix, OVector, SMatrix, U3, U4, U6};

pub type Alg6 = OMatrix<f64, U4, U4>;
pub type Grp6 = OMatrix<f64, U4, U4>;
pub type Vec6 = OVector<f64, U6>;

#[derive(Debug, Clone, Copy)]
pub enum SE3 {
    Grp(Grp6),
    Alg(Alg6),
    Vec(Vec6),
}

impl SE3 {
    pub fn from_grp(grp: Grp6) -> Self {
        Self::Grp(grp)
    }
    pub fn from_alg(alg: Alg6) -> Self {
        Self::Alg(alg)
    }
    pub fn from_vec(vec: Vec6) -> Self {
        Self::Vec(vec)
    }
}

pub fn hat(τ: Vec6) -> Alg6 {
    let (ρ, θ) = decombine(τ);
    let mut alg = Alg6::zeros();
    let θ_alg = hat3(θ);
    alg.index_mut((0..3, 0..3)).copy_from(&θ_alg);
    alg.index_mut((0..3, 3)).copy_from(&ρ);

    alg
}

pub fn vee(alg: Alg6) -> Vec6 {
    let θ_alg: Alg3 = alg.fixed_slice::<3, 3>(0, 0).into();
    let ρ: Vec3 = alg.fixed_slice::<3, 1>(0, 3).into();

    let θ = vee3(θ_alg);

    combine(ρ, θ)
}

pub fn combine(ρ: Vec3, θ: Vec3) -> Vec6 {
    Vec6::new(ρ[0], ρ[1], ρ[2], θ[0], θ[1], θ[2])
}

pub fn decombine(τ: Vec6) -> (Vec3, Vec3) {
    let ρ = Vec3::new(τ[0], τ[1], τ[2]);
    let θ = Vec3::new(τ[3], τ[4], τ[5]);
    (ρ, θ)
}

pub fn exp(τ_alg: Alg6) -> Grp6 {
    let τ = vee(τ_alg);
    let (ρ, θ_vec) = decombine(τ);
    let θ_alg: Alg3 = τ_alg.fixed_slice::<3, 3>(0, 0).into();
    let θ_grp = exp3(θ_alg);

    let θ = (θ_vec.dot(&θ_vec)).sqrt();
    let (a, b) = if θ < SMALL_FLOAT {
        let a = 0.5 - θ.powi(2) / 24.0 + θ.powi(4) / 720.0;
        let b = 1.0 / 6.0 - θ.powi(2) / 120.0 + θ.powi(4) / 5040.0;
        (a, b)
    } else {
        let a = (1.0 - θ.cos()) / θ.powi(2);
        let b = (θ - θ.sin()) / θ.powi(3);
        (a, b)
    };
    let v_m = OMatrix::<f64, U3, U3>::identity() + a * θ_alg + b * θ_alg.pow(2);
    let t = v_m * ρ;

    let mut grp = Grp6::zeros();
    grp.index_mut((0..3, 0..3)).copy_from(&θ_grp);
    grp.index_mut((0..3, 3)).copy_from(&t);
    grp[(3, 3)] = 1.0;

    grp
}

pub fn log(grp: Grp6) -> Alg6 {
    let θ_grp: Grp3 = grp.fixed_slice::<3, 3>(0, 0).into();
    let t: Vec3 = grp.fixed_slice::<3, 1>(0, 3).into();

    let θ_alg = log3(θ_grp);

    let θ_vec = vee3(θ_alg);
    let θ = (θ_vec.dot(&θ_vec)).sqrt();
    let a = if θ < SMALL_FLOAT {
        let a = (1.0 / 12.0 - θ.powi(2) / 180.0 + θ.powi(4) / 5040.0)
            / (1.0 - θ.powi(2) / 12.0 + θ.powi(4) / 360.0);
        a
    } else {
        let a = (1.0 - θ * θ.sin() / 2.0 / (1.0 - θ.cos())) / θ.powi(2);
        a
    };
    let vm_inv = OMatrix::<f64, U3, U3>::identity() - 0.5 * θ_alg + a * θ_alg.pow(2);
    let ρ = vm_inv * t;

    let mut τ_alg = Alg6::zeros();
    τ_alg.index_mut((0..3, 0..3)).copy_from(&θ_alg);
    τ_alg.index_mut((0..3, 3)).copy_from(&ρ);

    τ_alg
}
