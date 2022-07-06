/// Generators of SO(3) are used as below:
///
/// E1 =    |0  0   0 |
///         |0  0   -1|
///         |0  1   0 |
///
/// E2 =    |0  0   1 |
///         |0  0   0 |
///         |-1 0   0 |
///
/// E3 =    |0  -1  0 |
///         |1  0   0 |
///         |0  0   0 |
///

///
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

pub const SMALL_FLOAT: f64 = 1e-7;

pub type Alg3 = OMatrix<f64, U3, U3>;
pub type Grp3 = OMatrix<f64, U3, U3>;
pub type Vec3 = OVector<f64, U3>;

/// Enum for SO(3) element.
/// SO(3) element can be expressed in three forms,i.e. in group, in algebra, and in vector.
#[derive(Debug, Clone, Copy)]
pub enum SO3 {
    Grp(Grp3),
    Alg(Alg3),
    Vec(Vec3),
}
impl SO3 {
    /// create an SO(3) element from group
    pub fn from_grp(grp: Grp3) -> Self {
        Self::Grp(grp)
    }
    /// create an SO(3) element from algebra
    pub fn from_alg(alg: Alg3) -> Self {
        Self::Alg(alg)
    }
    /// create an SO(3) element from vector
    pub fn from_vec(vec: Vec3) -> Self {
        Self::Vec(vec)
    }
}

/// SO(3) mapping vector to algebra
pub fn hat(w: Vec3) -> Alg3 {
    Alg3::new(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)
}
/// SO(3) mapping algebra to vector
pub fn vee(alg: Alg3) -> Vec3 {
    Vec3::new(alg.m32, alg.m13, alg.m21)
}

/// SO(3) mapping algebra to group
pub fn exp(alg: Alg3) -> Grp3 {
    let θ_vec = Vec3::new(alg.m32, alg.m13, alg.m21);

    let θ = (θ_vec.dot(&θ_vec)).sqrt();
    let (a, b) = if θ < SMALL_FLOAT {
        let a = 1.0 - θ.powi(2) / 6.0 + θ.powi(4) / 120.0;
        let b = 0.5 - θ.powi(2) / 24.0 + θ.powi(4) / 720.0;
        (a, b)
    } else {
        let a = θ.sin() / θ;
        let b = (1.0 - θ.cos()) / θ.powi(2);
        (a, b)
    };
    Grp3::identity() + a * &alg + b * &alg.pow(2)
}

/// SO(3) mapping group to algebra
pub fn log(grp: Grp3) -> Alg3 {
    let trace = grp.trace();
    if (trace - 3.0).abs() < SMALL_FLOAT {
        // θ = 0
        let θ = 0.0;
        let w = OVector::<f64, U3>::zeros();
        Alg3::zeros()
    } else if (trace + 1.0).abs() < SMALL_FLOAT {
        // θ = π
        let θ = PI;
        let diag = grp.diagonal();

        let (k, mx) = diag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let v = grp.column(k) + Grp3::identity().column(k);
        let u = v / (2.0 * (1.0 + mx)).sqrt();
        let θ_vec = θ * u;
        let θ_alg = hat(θ_vec);
        θ_alg
    } else {
        // general case
        let d_r = grp - &grp.transpose();
        let θ = ((trace - 1.0) / 2.0).acos();
        let a = θ / 2.0 / θ.sin();
        let mut θ_alg = a * d_r;
        if (exp(θ_alg) - grp).norm() >= SMALL_FLOAT {
            θ_alg = -θ_alg;
        }
        θ_alg
    }
}

/// SO(3) get right jacobian matrix
pub fn jac_r(θ_vec: Vec3) -> OMatrix<f64, U3, U3> {
    let θ = (θ_vec.dot(&θ_vec)).sqrt();
    let θ_alg = hat(θ_vec);
    let (a, b) = if θ < SMALL_FLOAT {
        let a = -0.5 + θ.powi(2) / 24.0 - θ.powi(4) / 720.0;
        let b = 1.0 / 6.0 - θ.powi(2) / 120.0 + θ.powi(4) / 5040.0;
        (a, b)
    } else {
        let a = (θ.cos() - 1.0) / θ.powi(2);
        let b = (θ - θ.sin()) / θ.powi(3);
        (a, b)
    };
    OMatrix::<f64, U3, U3>::identity() + a * θ_alg + b * θ_alg.pow(2)
}

/// a trait for transforming the SO(3) element from one form to another
pub trait One2OneMap {
    fn to_grp(self) -> Grp3;
    fn to_alg(self) -> Alg3;
    fn to_vec(self) -> Vec3;
}

impl One2OneMap for SO3 {
    /// transforming the SO(3) element to the form of algebra
    fn to_alg(self) -> Alg3 {
        match self {
            Self::Alg(alg) => alg,
            Self::Grp(grp) => log(grp),
            Self::Vec(vec) => hat(vec),
        }
    }
    /// transforming the SO(3) element to the form of group
    fn to_grp(self) -> Grp3 {
        match self {
            Self::Alg(alg) => exp(alg),
            Self::Grp(grp) => grp,
            Self::Vec(vec) => exp(hat(vec)),
        }
    }
    /// transforming the SO(3) element to the form of vector
    fn to_vec(self) -> Vec3 {
        match self {
            Self::Alg(alg) => vee(alg),
            Self::Grp(grp) => vee(log(grp)),
            Self::Vec(vec) => vec,
        }
    }
}

impl SO3 {
    /// inverse the SO(3) element
    pub fn inverse(&self) -> Self {
        let r_inv = self.to_grp().transpose();
        Self::from_grp(r_inv)
    }
    /// adjoint matrix of the SO(3) element
    pub fn adj(&self) -> Grp3 {
        self.to_grp()
    }
    /// for SO(3) element, action on vector
    pub fn act_v(&self, x: Vec3) -> Vec3 {
        self.to_grp() * x
    }
    /// for SO(3) element, action on element
    pub fn act_g(&self, x: Self) -> Self {
        Self::from_grp(self.to_grp() * x.to_grp())
    }
    /// SO(3) element right plus a vector
    pub fn plus_r(&self, x: Vec3) -> Self {
        let so2 = SO3::from_vec(x);
        self.act_g(so2)
    }
    /// SO(3) element right minus another element
    pub fn minus_r(&self, x: Self) -> Vec3 {
        let dso = x.inverse().act_g(*self);
        dso.to_vec()
    }
}

pub mod sosekf {
    use crate::time::{Duration, Epoch};

    use super::{hat, jac_r, Alg3, Grp3, One2OneMap, Vec3, SO3};
    use crate::alloc::{boxed::Box, vec::Vec};
    use crate::errors::YakfError;

    use crate::linalg::allocator::Allocator;
    use crate::linalg::{Const, DefaultAllocator, DimName, OMatrix, OVector, U3, U4, U6};
    pub struct SOEKF {
        pub state: SO3,
        pmatrix: OMatrix<f64, U3, U3>,
        qmatrix: OMatrix<f64, U3, U3>,
        nmatrix: OMatrix<f64, Const<12>, Const<12>>,
    }
    impl SOEKF {
        #[allow(dead_code)]
        /// function that returns a UKF
        pub fn build(
            state: SO3,
            pmatrix: OMatrix<f64, U3, U3>,
            qmatrix: OMatrix<f64, U3, U3>,
            nmatrix: OMatrix<f64, Const<12>, Const<12>>,
        ) -> Self {
            Self {
                state,
                pmatrix,
                qmatrix,
                nmatrix,
            }
        }

        pub fn transition_f(&self, &u: &Vec3, dt: Duration) -> Grp3 {
            let v = u * dt.in_seconds();
            let so = SO3::from_vec(v);
            let x = so.inverse().adj();
            x
        }

        pub fn transition_g(&self, u: &Vec3, dt: Duration) -> OMatrix<f64, U3, U3> {
            let v = u * dt.in_seconds();
            jac_r(v)
        }

        pub fn transition_h(&self, x_predict: &SO3, bk: &[Vec3; 4]) -> OMatrix<f64, Const<12>, U3> {
            let mut m = OMatrix::<f64, Const<12>, U3>::zeros();
            for i in 0..4 {
                let r = x_predict.to_grp();
                let r_t = r.transpose();
                let left = -r_t * hat(bk[i]);
                let right = -r;
                let block = left * right;
                m.index_mut((i * 3..i * 3 + 3, 0..3)).copy_from(&block);
            }
            m
        }

        pub fn propagate(&self, u: &Vec3, dt: Duration) -> SO3 {
            let v = u * dt.in_seconds();
            let y = self.state.plus_r(v);
            y
        }
        pub fn measure(&self, x_predict: &SO3, bk: &[Vec3; 4]) -> OVector<f64, Const<12>> {
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
            u: Vec3,
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
