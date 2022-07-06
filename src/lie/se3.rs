use super::so3::{
    exp as exp3, hat as hat3, jac_r as jac3_r, log as log3, vee as vee3, Alg3, Grp3, Vec3,
    SMALL_FLOAT, SO3,
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

///
///
/// ///
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

pub fn jac_r(ρ: Vec3, θ_vec: Vec3) -> OMatrix<f64, U6, U6> {
    jac_l(-ρ, -θ_vec)
}

pub trait One2OneMapSE {
    fn to_grp(self) -> Grp6;
    fn to_alg(self) -> Alg6;
    fn to_vec(self) -> Vec6;
}

impl One2OneMapSE for SE3 {
    fn to_alg(self) -> Alg6 {
        match self {
            Self::Alg(alg) => alg,
            Self::Grp(grp) => log(grp),
            Self::Vec(vec) => hat(vec),
        }
    }
    fn to_grp(self) -> Grp6 {
        match self {
            Self::Alg(alg) => exp(alg),
            Self::Grp(grp) => grp,
            Self::Vec(vec) => exp(hat(vec)),
        }
    }
    fn to_vec(self) -> Vec6 {
        match self {
            Self::Alg(alg) => vee(alg),
            Self::Grp(grp) => vee(log(grp)),
            Self::Vec(vec) => vec,
        }
    }
}

impl SE3 {
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
    pub fn act_v(&self, x: Vec3) -> Vec3 {
        let τ_grp = self.to_grp();
        let r: Grp3 = τ_grp.fixed_slice::<3, 3>(0, 0).into();
        let t: Vec3 = τ_grp.fixed_slice::<3, 1>(0, 3).into();
        t + r * x
    }
    pub fn act_g(&self, x: Self) -> Self {
        Self::from_grp(self.to_grp() * x.to_grp())
    }
    pub fn plus_r(&self, x: Vec6) -> Self {
        let se2 = SE3::from_vec(x);
        self.act_g(se2)
    }
    pub fn minus_r(&self, x: Self) -> Vec6 {
        let dse = x.inverse().act_g(*self);
        dse.to_vec()
    }

    pub fn to_r_t(&self) -> (Grp3, Vec3) {
        let τ_grp = self.to_grp();
        let r: Grp3 = τ_grp.fixed_slice::<3, 3>(0, 0).into();
        let t: Vec3 = τ_grp.fixed_slice::<3, 1>(0, 3).into();
        (r, t)
    }

    pub fn from_r_t(r: Grp3, t: Vec3) -> Self {
        let mut grp = Grp6::zeros();
        grp.index_mut((0..3, 0..3)).copy_from(&r);
        grp.index_mut((0..3, 3)).copy_from(&t);
        grp[(3, 3)] = 1.0;
        Self::from_grp(grp)
    }
}

pub mod seekf {
    use crate::time::{Duration, Epoch};

    use super::{combine, decombine, hat, hat3, jac_r, Alg6, Grp6, One2OneMapSE, Vec3, Vec6, SE3};
    use crate::alloc::{boxed::Box, vec::Vec};
    use crate::errors::YakfError;

    use crate::linalg::allocator::Allocator;
    use crate::linalg::{Const, DefaultAllocator, DimName, OMatrix, OVector, U3, U4, U6};
    pub struct SEEKF {
        pub state: SE3,
        pmatrix: OMatrix<f64, U6, U6>,
        qmatrix: OMatrix<f64, U6, U6>,
        nmatrix: OMatrix<f64, Const<12>, Const<12>>,
    }
    impl SEEKF {
        #[allow(dead_code)]
        /// function that returns a UKF
        pub fn build(
            state: SE3,
            pmatrix: OMatrix<f64, U6, U6>,
            qmatrix: OMatrix<f64, U6, U6>,
            nmatrix: OMatrix<f64, Const<12>, Const<12>>,
        ) -> Self {
            Self {
                state,
                pmatrix,
                qmatrix,
                nmatrix,
            }
        }

        pub fn transition_f(&self, &u: &Vec6, dt: Duration) -> OMatrix<f64, U6, U6> {
            let v = u * dt.in_seconds();
            let so = SE3::from_vec(v);
            let x = so.inverse().adj();
            x
        }

        pub fn transition_g(&self, u: &Vec6, dt: Duration) -> OMatrix<f64, U6, U6> {
            let v = u * dt.in_seconds();
            let (ρ, θ) = decombine(v);

            jac_r(ρ, θ)
        }

        pub fn transition_h(&self, x_predict: &SE3, bk: &[Vec3; 4]) -> OMatrix<f64, Const<12>, U6> {
            let mut m = OMatrix::<f64, Const<12>, U6>::zeros();
            let (r, t) = x_predict.to_r_t();
            let r_t = r.transpose();
            for i in 0..4 {
                let mut left = OMatrix::<f64, U3, U6>::zeros();
                left.index_mut((0..3, 0..3)).copy_from(&r);
                left.index_mut((0..3, 3..6)).copy_from(&(-r * hat3(bk[i])));
                let right = -x_predict.adj();
                let block = left * right;
                m.index_mut((i * 3..i * 3 + 3, 0..6)).copy_from(&block);
            }
            m
        }

        pub fn propagate(&self, u: &Vec6, dt: Duration) -> SE3 {
            let v = u * dt.in_seconds();
            let y = self.state.plus_r(v);
            y
        }
        pub fn measure(&self, x_predict: &SE3, bk: &[Vec3; 4]) -> OVector<f64, Const<12>> {
            let mut ob = OVector::<f64, Const<12>>::zeros();
            let x_inv = x_predict.inverse();
            for i in 0..4 {
                let block = x_inv.act_v(bk[i]);
                ob.index_mut((i * 3..i * 3 + 3, 0..1)).copy_from(&block);
            }
            ob
        }

        #[allow(dead_code)]
        pub fn feed_and_update(
            &mut self,
            measure: OVector<f64, Const<12>>,
            dt: Duration,
            u: Vec6,
            bk: &[Vec3; 4],
        ) -> Result<(), YakfError> {
            let mut x_predict = self.propagate(&u, dt);

            let f = self.transition_f(&u, dt);
            let g = self.transition_g(&u, dt);

            let p_predict = f * &self.pmatrix * &f.transpose() + g * &self.qmatrix * &g.transpose();

            let ob_predict = self.measure(&x_predict, bk);

            let z = measure - ob_predict;

            let h = self.transition_h(&x_predict, bk);

            let zmatrix = h * p_predict * h.transpose() + self.nmatrix;

            match zmatrix.try_inverse() {
                Some(zm_inv) => {
                    let kmatrix = p_predict * h.transpose() * zm_inv;
                    let dx = kmatrix * z;
                    self.state = x_predict.plus_r(dx);
                    self.pmatrix = &self.pmatrix - &kmatrix * &zmatrix * &kmatrix.transpose();

                    Ok(())
                }
                None => Err(YakfError::InverseErr),
            }
            // let k = p_predict*h.transpose()*
            // let g_x = self.transition_f(&x_predict, dt);
            // let p_predict = &g_x * &self.prev_p * g_x.transpose() + &self.process_q;

            // let h_x = self.transition_h(&x_predict);
            // match (&h_x * &p_predict * &h_x.transpose() + &self.process_r).try_inverse() {
            //     Some(inv) => {
            //         let k = &p_predict * h_x.transpose() * inv;
            //         let new_estimate = x_predict + &k * (measure - z_predict);
            //         self.prev_x.set_state(new_estimate);
            //         self.prev_x.set_epoch(m_epoch);
            //         let sub = dmatrix_identity(self.n, self.n) - &k * h_x;
            //         self.prev_p = &sub * &p_predict * &sub.transpose() + &self.process_r;
            //         Ok(())
            //     }
            //     None => Err(YakfError::InverseErr),
            // }
        }
    }
}
