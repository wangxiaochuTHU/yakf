/// Generators of SO(2) is used as below:
///
/// E =    |0  -1|
///        |1   0|

///
use nalgebra::ComplexField;

use crate::linalg::{OMatrix, OVector, U2};
use libm::atan2;

// use super::constants::SMALL_FLOAT;

pub type Alg1 = OMatrix<f64, U2, U2>;
pub type Grp1 = OMatrix<f64, U2, U2>;
pub type Vec1 = f64;
pub type Vec2 = OVector<f64, U2>;

/// Enum for SO(2) element.
/// SO(2) element can be expressed in three forms,i.e. in group, in algebra, and in vector.
#[derive(Debug, Clone, Copy)]
pub enum SO2 {
    Grp(Grp1),
    Alg(Alg1),
    Vec(Vec1),
}
impl SO2 {
    /// create an SO(2) element from group
    pub fn from_grp(grp: Grp1) -> Self {
        Self::Grp(grp)
    }
    /// create an SO(2) element from algebra
    pub fn from_alg(alg: Alg1) -> Self {
        Self::Alg(alg)
    }
    /// create an SO(2) element from vector
    pub fn from_vec(vec: Vec1) -> Self {
        Self::Vec(vec)
    }
}

/// SO(2) mapping vector to algebra
pub fn hat(w: Vec1) -> Alg1 {
    Alg1::new(0.0, -w, w, 0.0)
}
/// SO(2) mapping algebra to vector
pub fn vee(alg: Alg1) -> Vec1 {
    alg.m21
}

/// SO(2) mapping algebra to group
pub fn exp(alg: Alg1) -> Grp1 {
    let θ = alg.m21;
    let cosθ = θ.cos();
    let sinθ = θ.sin();

    Grp1::new(cosθ, -sinθ, sinθ, cosθ)
}

/// SO(2) mapping group to algebra
pub fn log(grp: Grp1) -> Alg1 {
    let θ = atan2(grp[(1, 0)], grp[(0, 0)]);
    hat(θ)
}

/// SO(2) mapping group to vec1
#[allow(non_snake_case)]
pub fn Log(grp: Grp1) -> Vec1 {
    atan2(grp[(1, 0)], grp[(0, 0)])
}

/// SO(2) get right jacobian matrix
pub fn jac_r(_θ: Vec1) -> f64 {
    1.0
}

/// a trait for transforming the SO(3) element from one form to another
pub trait One2OneMapSO2 {
    fn to_grp(self) -> Grp1;
    fn to_alg(self) -> Alg1;
    fn to_vec(self) -> Vec1;
}

impl One2OneMapSO2 for SO2 {
    /// transforming the SO(2) element to the form of algebra
    fn to_alg(self) -> Alg1 {
        match self {
            Self::Alg(alg) => alg,
            Self::Grp(grp) => log(grp),
            Self::Vec(vec) => hat(vec),
        }
    }
    /// transforming the SO(2) element to the form of group
    fn to_grp(self) -> Grp1 {
        match self {
            Self::Alg(alg) => exp(alg),
            Self::Grp(grp) => grp,
            Self::Vec(vec) => exp(hat(vec)),
        }
    }
    /// transforming the SO(2) element to the form of vector
    fn to_vec(self) -> Vec1 {
        match self {
            Self::Alg(alg) => vee(alg),
            Self::Grp(grp) => vee(log(grp)),
            Self::Vec(vec) => vec,
        }
    }
}

impl SO2 {
    /// inverse the SO(2) element
    pub fn inverse(&self) -> Self {
        let r_inv = self.to_grp().transpose();
        Self::from_grp(r_inv)
    }
    /// adjoint matrix of the SO(2) element
    pub fn adj(&self) -> f64 {
        1.0
    }
    /// for SO(2) element, action on vector
    pub fn act_v(&self, x: Vec2) -> Vec2 {
        self.to_grp() * x
    }
    /// for SO(2) element, action on element
    pub fn act_g(&self, x: Self) -> Self {
        Self::from_grp(self.to_grp() * x.to_grp())
    }
    /// SO(2) element right plus a vector
    pub fn plus_r(&self, x: Vec1) -> Self {
        let so2 = SO2::from_vec(x);
        self.act_g(so2)
    }
    /// SO(2) element right minus another element
    pub fn minus_r(&self, x: Self) -> Vec1 {
        let dso = x.inverse().act_g(*self);
        dso.to_vec()
    }
}
