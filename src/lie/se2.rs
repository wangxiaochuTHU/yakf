use super::constants::SMALL_FLOAT;
use super::so2::{exp as exp1, hat as hat1, log as log1, vee as vee1, Alg1, Grp1, Vec1, Vec2};
/// Generators of SE(2) are used as below:
///
/// E1 =    |0   0   1|
///         |0   0   0|
///         |0   0   0|
///
/// E2 =    |0   0   0|
///         |0   0   1|
///         |0   0   0|
///
/// E3 =    |0  0   0   0|
///         |0  0   0   0|
///         |0  0   0   1|
///         |0  0   0   0|
///
/// E4 =    |0  -1  0|
///         |1  0   0|
///         |0  0   0|
///

///  se(2) vector follows the order: [ ρ , θ ]
use nalgebra::ComplexField;

use crate::linalg::{OMatrix, OVector, U2, U3};

pub type Alg3 = OMatrix<f64, U3, U3>;
pub type Grp3 = OMatrix<f64, U3, U3>;
pub type Vec3 = OVector<f64, U3>;

///
///
/// Enum for SE(2) element.
/// SE(2) element can be expressed in three forms,i.e. in group, in algebra, and in vector.
#[derive(Debug, Clone, Copy)]
pub enum SE2 {
    Grp(Grp3),
    Alg(Alg3),
    Vec(Vec3),
}

impl SE2 {
    /// create an SE(2) element from group
    pub fn from_grp(grp: Grp3) -> Self {
        Self::Grp(grp)
    }
    /// create an SE(2) element from algebra
    pub fn from_alg(alg: Alg3) -> Self {
        Self::Alg(alg)
    }
    /// create an SE(2) element from vector
    pub fn from_vec(vec: Vec3) -> Self {
        Self::Vec(vec)
    }
}

/// SE(2) mapping vector to algebra
pub fn hat(τ: Vec3) -> Alg3 {
    let (ρ, θ) = decombine(τ);
    let mut alg = Alg3::zeros();
    let θ_alg = hat1(θ);
    alg.index_mut((0..2, 0..2)).copy_from(&θ_alg);
    alg.index_mut((0..2, 2)).copy_from(&ρ);

    alg
}

/// SE(2) mapping algebra to vector
pub fn vee(alg: Alg3) -> Vec3 {
    let θ_alg: Alg1 = alg.fixed_slice::<2, 2>(0, 0).into();
    let ρ: Vec2 = alg.fixed_slice::<2, 1>(0, 2).into();

    let θ = vee1(θ_alg);

    combine(ρ, θ)
}

/// combine vector2 and scalar (ρ θ) into one vector3 (τ)
pub fn combine(ρ: Vec2, θ: Vec1) -> Vec3 {
    Vec3::new(ρ[0], ρ[1], θ)
}

/// decombine a vector3 τ into vector2 and scalar, (ρ θ)
pub fn decombine(τ: Vec3) -> (Vec2, Vec1) {
    let ρ = Vec2::new(τ[0], τ[1]);
    let θ = τ[2];
    (ρ, θ)
}

/// SE(2) mapping algebra to group
pub fn exp(τ_alg: Alg3) -> Grp3 {
    let τ = vee(τ_alg);
    let (ρ, θ) = decombine(τ);

    let (a, b) = if θ < SMALL_FLOAT {
        let a = 1.0 - θ.powi(2) / 6.0 + θ.powi(4) / 120.0;
        let b = θ / 2.0 - θ.powi(3) / 24.0 + θ.powi(5) / 720.0;
        (a, b)
    } else {
        let a = θ.sin() / θ;
        let b = (1.0 - θ.cos()) / θ;
        (a, b)
    };
    let θ_grp = exp1(hat1(θ));
    let v_m = OMatrix::<f64, U2, U2>::new(a, -b, b, a);
    let t = v_m * ρ;

    let mut grp = Grp3::zeros();
    grp.index_mut((0..2, 0..2)).copy_from(&θ_grp);
    grp.index_mut((0..2, 2)).copy_from(&t);
    grp[(2, 2)] = 1.0;

    grp
}

/// SE(2) mapping vec3 to group
#[allow(non_snake_case)]
pub fn Exp(τ: Vec3) -> Grp3 {
    let (ρ, θ) = decombine(τ);

    let (a, b) = if θ < SMALL_FLOAT {
        let a = 1.0 - θ.powi(2) / 6.0 + θ.powi(4) / 120.0;
        let b = θ / 2.0 - θ.powi(3) / 24.0 + θ.powi(5) / 720.0;
        (a, b)
    } else {
        let a = θ.sin() / θ;
        let b = (1.0 - θ.cos()) / θ;
        (a, b)
    };
    let θ_grp = exp1(hat1(θ));
    let v_m = OMatrix::<f64, U2, U2>::new(a, -b, b, a);
    let t = v_m * ρ;

    let mut grp = Grp3::zeros();
    grp.index_mut((0..2, 0..2)).copy_from(&θ_grp);
    grp.index_mut((0..2, 2)).copy_from(&t);
    grp[(2, 2)] = 1.0;

    grp
}

/// SE(2) mapping group to algebra
pub fn log(grp: Grp3) -> Alg3 {
    let θ_grp: Grp1 = grp.fixed_slice::<2, 2>(0, 0).into();
    let t: Vec2 = grp.fixed_slice::<2, 1>(0, 2).into();

    let θ_alg = log1(θ_grp);
    let θ = vee1(θ_alg);

    let (a, b) = if θ < SMALL_FLOAT {
        let a = 1.0 - θ.powi(2) / 6.0 + θ.powi(4) / 120.0;
        let b = θ / 2.0 - θ.powi(3) / 24.0 + θ.powi(5) / 720.0;
        (a, b)
    } else {
        let a = θ.sin() / θ;
        let b = (1.0 - θ.cos()) / θ;
        (a, b)
    };

    let vm_inv = OMatrix::<f64, U2, U2>::new(a, b, -b, a) / (a.powi(2) + b.powi(2));
    let ρ = vm_inv * t;

    let τ = combine(ρ, θ);
    let τ_alg = hat(τ);

    τ_alg
}

/// SE(2) mapping group to vec3
#[allow(non_snake_case)]
pub fn Log(grp: Grp3) -> Vec3 {
    let θ_grp: Grp1 = grp.fixed_slice::<2, 2>(0, 0).into();
    let t: Vec2 = grp.fixed_slice::<2, 1>(0, 2).into();

    let θ_alg = log1(θ_grp);
    let θ = vee1(θ_alg);

    let (a, b) = if θ < SMALL_FLOAT {
        let a = 1.0 - θ.powi(2) / 6.0 + θ.powi(4) / 120.0;
        let b = θ / 2.0 - θ.powi(3) / 24.0 + θ.powi(5) / 720.0;
        (a, b)
    } else {
        let a = θ.sin() / θ;
        let b = (1.0 - θ.cos()) / θ;
        (a, b)
    };

    let vm_inv = OMatrix::<f64, U2, U2>::new(a, b, -b, a) / (a.powi(2) + b.powi(2));
    let ρ = vm_inv * t;

    let τ = combine(ρ, θ);

    τ
}

/// TODO ： CHECK for left jac and right jac conversion
pub fn jac_l(ρ: Vec2, θ: Vec1) -> OMatrix<f64, U3, U3> {
    let (a, b) = if θ < SMALL_FLOAT {
        let a = 1.0 - θ.powi(2) / 6.0 + θ.powi(4) / 120.0;
        let b = θ / 2.0 - θ.powi(3) / 24.0 + θ.powi(5) / 720.0;
        (a, b)
    } else {
        let a = θ.sin() / θ;
        let b = (1.0 - θ.cos()) / θ;
        (a, b)
    };

    let (c, d) = if θ < SMALL_FLOAT {
        let c = ρ[0] * (θ / 6.0 - θ.powi(3) / 120.0)
            + ρ[1] * (0.5 - θ.powi(2) / 24.0 + θ.powi(4) / 720.0);
        let d = ρ[1] * (θ / 6.0 - θ.powi(3) / 120.0)
            - ρ[0] * (0.5 - θ.powi(2) / 24.0 + θ.powi(4) / 720.0);
        (c, d)
    } else {
        let c = (ρ[0] * (θ - θ.sin()) + ρ[1] * (1.0 - θ.cos())) / θ.powi(2);
        let d = (ρ[1] * (θ - θ.sin()) - ρ[0] * (1.0 - θ.cos())) / θ.powi(2);
        (c, d)
    };

    let jacl = OMatrix::<f64, U3, U3>::new(a, -b, c, b, a, d, 0.0, 0.0, 1.0);

    jacl
}

/// SE(2) get right jacobian matrix
pub fn jac_r(ρ: Vec2, θ: Vec1) -> OMatrix<f64, U3, U3> {
    let (a, b) = if θ < SMALL_FLOAT {
        let a = 1.0 - θ.powi(2) / 6.0 + θ.powi(4) / 120.0;
        let b = θ / 2.0 - θ.powi(3) / 24.0 + θ.powi(5) / 720.0;
        (a, b)
    } else {
        let a = θ.sin() / θ;
        let b = (1.0 - θ.cos()) / θ;
        (a, b)
    };

    let (c, d) = if θ < SMALL_FLOAT {
        let c = ρ[0] * (θ / 6.0 - θ.powi(3) / 120.0)
            - ρ[1] * (0.5 - θ.powi(2) / 24.0 + θ.powi(4) / 720.0);
        let d = ρ[1] * (θ / 6.0 - θ.powi(3) / 120.0)
            + ρ[0] * (0.5 - θ.powi(2) / 24.0 + θ.powi(4) / 720.0);
        (c, d)
    } else {
        let c = (ρ[0] * (θ - θ.sin()) - ρ[1] * (1.0 - θ.cos())) / θ.powi(2);
        let d = (ρ[1] * (θ - θ.sin()) + ρ[0] * (1.0 - θ.cos())) / θ.powi(2);
        (c, d)
    };

    let jacr = OMatrix::<f64, U3, U3>::new(a, b, c, -b, a, d, 0.0, 0.0, 1.0);

    jacr
}

/// a trait for transforming the SE(2) element from one form to another
pub trait One2OneMapSE2 {
    fn to_grp(self) -> Grp3;
    fn to_alg(self) -> Alg3;
    fn to_vec(self) -> Vec3;
}

impl One2OneMapSE2 for SE2 {
    /// transforming the SE(2) element to the form of algebra
    fn to_alg(self) -> Alg3 {
        match self {
            Self::Alg(alg) => alg,
            Self::Grp(grp) => log(grp),
            Self::Vec(vec) => hat(vec),
        }
    }
    /// transforming the SE(2) element to the form of group
    fn to_grp(self) -> Grp3 {
        match self {
            Self::Alg(alg) => exp(alg),
            Self::Grp(grp) => grp,
            Self::Vec(vec) => Exp(vec),
        }
    }
    /// transforming the SE(2) element to the form of vector
    fn to_vec(self) -> Vec3 {
        match self {
            Self::Alg(alg) => vee(alg),
            Self::Grp(grp) => Log(grp),
            Self::Vec(vec) => vec,
        }
    }
}

impl SE2 {
    // inverse the SE(2) element
    pub fn inverse(&self) -> Self {
        let τ_grp = self.to_grp();
        let θ_grp: Grp1 = τ_grp.fixed_slice::<2, 2>(0, 0).into();
        let t: Vec2 = τ_grp.fixed_slice::<2, 1>(0, 2).into();

        let r_inv = θ_grp.transpose();
        let new_θ_grp = r_inv;
        let new_t = -r_inv * t;

        let mut new_τ_grp = Grp3::zeros();
        new_τ_grp.index_mut((0..2, 0..2)).copy_from(&new_θ_grp);
        new_τ_grp.index_mut((0..2, 2)).copy_from(&new_t);
        new_τ_grp[(2, 2)] = 1.0;

        Self::from_grp(new_τ_grp)
    }

    /// adjoint matrix of the SE(2) element
    pub fn adj(&self) -> OMatrix<f64, U3, U3> {
        let τ_grp = self.to_grp();
        let θ_grp: Grp1 = τ_grp.fixed_slice::<2, 2>(0, 0).into();
        let t: Vec2 = τ_grp.fixed_slice::<2, 1>(0, 2).into();

        let mut ad = OMatrix::<f64, U3, U3>::zeros();
        ad.index_mut((0..2, 0..2)).copy_from(&θ_grp);
        ad.index_mut((0..3, 3..6)).copy_from(&(-hat1(1.0) * t));
        ad[(2, 2)] = 1.0;

        ad
    }

    /// for SE(2) element, action on vector
    pub fn act_v(&self, x: Vec2) -> Vec2 {
        let τ_grp = self.to_grp();
        let r: Grp1 = τ_grp.fixed_slice::<2, 2>(0, 0).into();
        let t: Vec2 = τ_grp.fixed_slice::<2, 1>(0, 2).into();
        t + r * x
    }

    /// for SE(2) element, action on element
    pub fn act_g(&self, x: Self) -> Self {
        Self::from_grp(self.to_grp() * x.to_grp())
    }

    /// SE(2) element right plus a vector
    pub fn plus_r(&self, x: Vec3) -> Self {
        let se2 = SE2::from_vec(x);
        self.act_g(se2)
    }

    /// SE(2) element right minus another element
    pub fn minus_r(&self, x: Self) -> Vec3 {
        let dse = x.inverse().act_g(*self);
        dse.to_vec()
    }

    /// rotation matrix and translation vector, extracted from a SE(2) element
    pub fn to_r_t(&self) -> (Grp1, Vec2) {
        let τ_grp = self.to_grp();
        let r: Grp1 = τ_grp.fixed_slice::<2, 2>(0, 0).into();
        let t: Vec2 = τ_grp.fixed_slice::<2, 1>(0, 2).into();
        (r, t)
    }

    /// create a SE(2) element, from a rotation matrix and a translation vector
    pub fn from_r_t(r: Grp1, t: Vec2) -> Self {
        let mut grp = Grp3::zeros();
        grp.index_mut((0..2, 0..2)).copy_from(&r);
        grp.index_mut((0..2, 2)).copy_from(&t);
        grp[(2, 2)] = 1.0;
        Self::from_grp(grp)
    }
}

mod tests {
    use super::{decombine, jac_l, jac_r, Vec1, Vec2, Vec3, SE2};
    #[test]
    fn test_jacr_jacl() {
        let τ = Vec3::new(0.3, 0.5, -0.2);
        let (ρ, θ) = decombine(τ);
        let se2 = SE2::from_vec(τ);
        let jacl = jac_l(ρ, θ);
        let jacr_on_expand = jac_r(ρ, θ);
        let jacr_on_convert = jac_l(-ρ, -θ);
        assert!((jacr_on_expand - jacr_on_convert).norm() < 1e-5);
    }
}
