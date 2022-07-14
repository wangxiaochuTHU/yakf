use super::constants::SMALL_FLOAT;
use super::so3::{
    exp as exp3, hat as hat3, jac_r as jac3_r, log as log3, vee as vee3, Alg3, Grp3, Vec3,
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

use crate::linalg::{OMatrix, OVector, U3, U4, U6};

pub type Alg6 = OMatrix<f64, U4, U4>;
pub type Grp6 = OMatrix<f64, U4, U4>;
pub type Vec6 = OVector<f64, U6>;

///
///
/// Enum for SE(3) element.
/// SE(3) element can be expressed in three forms,i.e. in group, in algebra, and in vector.
#[derive(Debug, Clone, Copy)]
pub enum SE3 {
    Grp(Grp6),
    Alg(Alg6),
    Vec(Vec6),
}

impl SE3 {
    /// create an SE(3) element from group
    pub fn from_grp(grp: Grp6) -> Self {
        Self::Grp(grp)
    }
    /// create an SE(3) element from algebra
    pub fn from_alg(alg: Alg6) -> Self {
        Self::Alg(alg)
    }
    /// create an SE(3) element from vector
    pub fn from_vec(vec: Vec6) -> Self {
        Self::Vec(vec)
    }
}

/// SE(3) mapping vector to algebra
pub fn hat(τ: Vec6) -> Alg6 {
    let (ρ, θ) = decombine(τ);
    let mut alg = Alg6::zeros();
    let θ_alg = hat3(θ);
    alg.index_mut((0..3, 0..3)).copy_from(&θ_alg);
    alg.index_mut((0..3, 3)).copy_from(&ρ);

    alg
}

/// SE(3) mapping algebra to vector
pub fn vee(alg: Alg6) -> Vec6 {
    let θ_alg: Alg3 = alg.fixed_slice::<3, 3>(0, 0).into();
    let ρ: Vec3 = alg.fixed_slice::<3, 1>(0, 3).into();

    let θ = vee3(θ_alg);

    combine(ρ, θ)
}

/// combine two vector3 (ρ θ) into one vector6 (τ)
pub fn combine(ρ: Vec3, θ: Vec3) -> Vec6 {
    Vec6::new(ρ[0], ρ[1], ρ[2], θ[0], θ[1], θ[2])
}

/// decombine a vector6 τ into two vector3 (ρ θ)
pub fn decombine(τ: Vec6) -> (Vec3, Vec3) {
    let ρ = Vec3::new(τ[0], τ[1], τ[2]);
    let θ = Vec3::new(τ[3], τ[4], τ[5]);
    (ρ, θ)
}

/// SE(3) mapping algebra to group
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

/// SE(3) mapping vec6 to group
#[allow(non_snake_case)]
pub fn Exp(τ: Vec6) -> Grp6 {
    let (ρ, θ_vec) = decombine(τ);
    let τ_alg = hat(τ);
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

/// SE(3) mapping group to algebra
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

/// SE(3) mapping group to vec6
#[allow(non_snake_case)]
pub fn Log(grp: Grp6) -> Vec6 {
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

    combine(ρ, θ_vec)
}

/// TODO: CHECK AGAIN
fn jlq(ρ: Vec3, θ_vec: Vec3) -> OMatrix<f64, U3, U3> {
    let θ = (θ_vec.dot(&θ_vec)).sqrt();
    let θ_alg = hat3(θ_vec);
    let ρ_alg = hat3(ρ);
    let a = 0.5;
    let ma = ρ_alg;
    let mb = &θ_alg * &ρ_alg + &ρ_alg * &θ_alg + &θ_alg * &ρ_alg * &θ_alg;
    let mc = &θ_alg.pow(2) * &ρ_alg + &ρ_alg * &θ_alg.pow(2) - 3.0 * &θ_alg * &ρ_alg * &θ_alg;
    let md = &θ_alg * &ρ_alg * &θ_alg.pow(2) + &θ_alg.pow(2) * &ρ_alg * &θ_alg;
    let (b, c, d) = if θ < SMALL_FLOAT {
        let b = 1.0 / 6.0 - θ.powi(2) / 120.0;
        let c = -1.0 / 24.0 + θ.powi(2) / 720.0;
        let d = 0.5 * (c - 3.0 * (-1.0 / 120.0 + θ.powi(2) / 5040.0));
        (b, c, d)
    } else {
        let b = (θ - θ.sin()) / θ.powi(3);
        let c = (1.0 - θ.powi(2) / 2.0 - θ.cos()) / θ.powi(4);
        let d = 0.5 * (c - 3.0 * (θ - θ.sin() - θ.powi(3) / 6.0) / θ.powi(5));
        (b, c, d)
    };
    a * ma + b * mb + c * mc + d * md
}

/// TODO ： CHECK for left jac and right jac conversion
pub fn jac_l(ρ: Vec3, θ_vec: Vec3) -> OMatrix<f64, U6, U6> {
    // let θ = (θ_vec.dot(&θ_vec)).sqrt();
    let jr3 = jac3_r(θ_vec);
    let jl3 = jr3.transpose(); // TODO: Check
    let jlq3 = jlq(ρ, θ_vec);

    let mut jacl = OMatrix::<f64, U6, U6>::zeros();
    jacl.index_mut((0..3, 0..3)).copy_from(&jl3);
    jacl.index_mut((0..3, 3..6)).copy_from(&jlq3);
    jacl.index_mut((3..6, 3..6)).copy_from(&jl3);

    jacl
}

/// SE(3) get right jacobian matrix
pub fn jac_r(ρ: Vec3, θ_vec: Vec3) -> OMatrix<f64, U6, U6> {
    jac_l(-ρ, -θ_vec)
}

/// a trait for transforming the SE(3) element from one form to another
pub trait One2OneMapSE {
    fn to_grp(self) -> Grp6;
    fn to_alg(self) -> Alg6;
    fn to_vec(self) -> Vec6;
}

impl One2OneMapSE for SE3 {
    /// transforming the SE(3) element to the form of algebra
    fn to_alg(self) -> Alg6 {
        match self {
            Self::Alg(alg) => alg,
            Self::Grp(grp) => log(grp),
            Self::Vec(vec) => hat(vec),
        }
    }
    /// transforming the SE(3) element to the form of group
    fn to_grp(self) -> Grp6 {
        match self {
            Self::Alg(alg) => exp(alg),
            Self::Grp(grp) => grp,
            Self::Vec(vec) => Exp(vec),
        }
    }
    /// transforming the SE(3) element to the form of vector
    fn to_vec(self) -> Vec6 {
        match self {
            Self::Alg(alg) => vee(alg),
            Self::Grp(grp) => Log(grp),
            Self::Vec(vec) => vec,
        }
    }
}

impl SE3 {
    // inverse the SE(3) element
    pub fn inverse(&self) -> Self {
        let τ_grp = self.to_grp();
        let θ_grp: Grp3 = τ_grp.fixed_slice::<3, 3>(0, 0).into();
        let t: Vec3 = τ_grp.fixed_slice::<3, 1>(0, 3).into();

        let r_inv = θ_grp.transpose();
        let new_θ_grp = r_inv;
        let new_t = -r_inv * t;

        let mut new_τ_grp = Grp6::zeros();
        new_τ_grp.index_mut((0..3, 0..3)).copy_from(&new_θ_grp);
        new_τ_grp.index_mut((0..3, 3)).copy_from(&new_t);
        new_τ_grp[(3, 3)] = 1.0;

        Self::from_grp(new_τ_grp)
    }

    /// adjoint matrix of the SE(3) element
    pub fn adj(&self) -> OMatrix<f64, U6, U6> {
        let τ_grp = self.to_grp();
        let θ_grp: Grp3 = τ_grp.fixed_slice::<3, 3>(0, 0).into();
        let t: Vec3 = τ_grp.fixed_slice::<3, 1>(0, 3).into();

        let mut ad = OMatrix::<f64, U6, U6>::zeros();
        ad.index_mut((0..3, 0..3)).copy_from(&θ_grp);
        ad.index_mut((0..3, 3..6)).copy_from(&(hat3(t) * θ_grp));
        ad.index_mut((3..6, 3..6)).copy_from(&θ_grp);

        ad
    }

    /// for SE(3) element, action on vector
    pub fn act_v(&self, x: Vec3) -> Vec3 {
        let τ_grp = self.to_grp();
        let r: Grp3 = τ_grp.fixed_slice::<3, 3>(0, 0).into();
        let t: Vec3 = τ_grp.fixed_slice::<3, 1>(0, 3).into();
        t + r * x
    }

    /// for SE(3) element, action on element
    pub fn act_g(&self, x: Self) -> Self {
        Self::from_grp(self.to_grp() * x.to_grp())
    }

    /// SE(3) element right plus a vector
    pub fn plus_r(&self, x: Vec6) -> Self {
        let se2 = SE3::from_vec(x);
        self.act_g(se2)
    }

    /// SE(3) element right minus another element
    pub fn minus_r(&self, x: Self) -> Vec6 {
        let dse = x.inverse().act_g(*self);
        dse.to_vec()
    }

    /// rotation matrix and translation vector, extracted from a SE(3) element
    pub fn to_r_t(&self) -> (Grp3, Vec3) {
        let τ_grp = self.to_grp();
        let r: Grp3 = τ_grp.fixed_slice::<3, 3>(0, 0).into();
        let t: Vec3 = τ_grp.fixed_slice::<3, 1>(0, 3).into();
        (r, t)
    }

    /// create a SE(3) element, from a rotation matrix and a translation vector
    pub fn from_r_t(r: Grp3, t: Vec3) -> Self {
        let mut grp = Grp6::zeros();
        grp.index_mut((0..3, 0..3)).copy_from(&r);
        grp.index_mut((0..3, 3)).copy_from(&t);
        grp[(3, 3)] = 1.0;
        Self::from_grp(grp)
    }
}
