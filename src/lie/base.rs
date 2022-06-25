use crate::time::{Duration, Epoch};
use alloc::borrow::ToOwned;
use core::convert::AsRef;
use core::convert::From;
use core::f64::consts::PI;

use crate::errors::YakfError;
use crate::linalg::allocator::Allocator;
use crate::linalg::{DefaultAllocator, DimName, OMatrix, OVector, SMatrix, U3, U4, U6};

const SMALL_FLOAT: f64 = 1e-6;
#[derive(Debug)]
pub struct LieGroupSE3 {
    // rotation matrix
    pub r: OMatrix<f64, U3, U3>,

    // translation vector
    pub t: OVector<f64, U3>,

    // in matrix form
    pub m: OMatrix<f64, U4, U4>,
}
impl<T: AsRef<LieVectorSE3>> From<T> for LieGroupSE3 {
    fn from(x: T) -> Self {
        x.as_ref().to_group()
    }
}
impl AsRef<LieVectorSE3> for LieVectorSE3 {
    fn as_ref(&self) -> &LieVectorSE3 {
        self
    }
}

impl LieGroupSE3 {
    pub fn to_algebra(&self) -> Option<LieVectorSE3> {
        let op_w = log3(&self.r);
        match op_w {
            // θ ≠ π
            Some(w) => {
                let theta = (w.dot(&w)).sqrt();
                let w_hat = hat3(&w);
                let (a, b) = if theta < SMALL_FLOAT {
                    let a = 1.0 - theta.powi(2) / 6.0 + theta.powi(4) / 120.0;
                    let b = 0.5 - theta.powi(2) / 24.0 + theta.powi(4) / 720.0;
                    (a, b)
                } else {
                    let a = theta.sin() / theta;
                    let b = (1.0 - theta.cos()) / theta.powi(2);
                    (a, b)
                };
                let v_inv = OMatrix::<f64, U3, U3>::identity() - 0.5 * &w_hat
                    + 1.0 / theta.powi(2) * (1.0 - a / b / 2.0) * w_hat.pow(2);
                let v = v_inv * self.t;
                Some(LieVectorSE3 { w: w, v: v })
            }

            //  θ = π
            None => None,
        }
    }

    /// calculate the Adjoint matrix, with the definition of vector column form as [rotation_vec, translation_vec]
    /// [ R      0
    ///  [t]xR   R  ]
    ///
    /// if the vector column form is defined as [translation_vec, rotation_vec], the Adjoint matrix should be
    /// [R   [t]xR;
    /// 0     R ]
    pub fn adjoint_matrix(&self) -> OMatrix<f64, U6, U6> {
        let mut adj = OMatrix::<f64, U6, U6>::zeros();
        adj.index_mut((0..3, 0..3)).copy_from(&self.r);
        let bottom_left_block = hat3(&self.t) * &self.r;
        adj.index_mut((3..6, 0..3)).copy_from(&bottom_left_block);

        adj.index_mut((3..6, 3..6)).copy_from(&self.r);

        adj
    }
    pub fn adjoint_action(&self, v: &OMatrix<f64, U4, U4>) -> Option<OMatrix<f64, U4, U4>> {
        match self.m.try_inverse() {
            Some(m_inv) => Some(self.m * v * m_inv),
            None => None,
        }
    }
    pub fn from_hat(m: &OMatrix<f64, U4, U4>) -> Self {
        let m = m.exp();
        let mut r = OMatrix::<f64, U3, U3>::zeros();
        let mut t = OVector::<f64, U3>::zeros();
        r.copy_from(&m.slice((0, 0), (3, 3)));
        t.copy_from(&m.slice((0, 3), (3, 1)));
        LieGroupSE3 { r: r, t: t, m: m }
    }
    pub fn to_hat(&self) -> Option<OMatrix<f64, U4, U4>> {
        match self.to_algebra() {
            Some(vec6) => Some(hat4(&vec6)),
            None => None,
        }
    }
}

#[derive(Debug)]
pub struct LieVectorSE3 {
    // w stands for rotation
    pub w: OVector<f64, U3>,

    // v stands for translation
    pub v: OVector<f64, U3>,
}

impl LieVectorSE3 {
    pub fn to_group(&self) -> LieGroupSE3 {
        let w_hat = hat3(&self.w);
        // method 1 to calculate `r` `t`
        let theta = (self.w.dot(&self.w)).sqrt();
        let m = if theta < SMALL_FLOAT {
            let a = 0.5 - theta.powi(2) / 24.0 + theta.powi(4) / 720.0;
            let b = -1.0 / 3.0 + theta.powi(2) / 30.0 - theta.powi(4) / 840.0;

            OMatrix::<f64, U3, U3>::identity() + a * &w_hat + b * w_hat.pow(2)
        } else {
            OMatrix::<f64, U3, U3>::identity()
                + (1.0 - theta.cos()) / theta.powi(2) * &w_hat
                + (theta - theta.sin()) / theta.powi(3) * w_hat.pow(2)
        };
        let r = exp3(&self.w);
        let t = m * &self.v;

        // method 2 to calculate `m`
        let mut m: OMatrix<f64, U4, U4> = OMatrix::<f64, U4, U4>::zeros();
        for (i, mut row) in m.row_iter_mut().enumerate() {
            if i < 3 {
                let x = w_hat.row(i);
                row.copy_from_slice(&[w_hat[(i, 0)], w_hat[(i, 1)], w_hat[(i, 2)], self.v[i]]);
            }
        }
        let m = m.exp();

        LieGroupSE3 { r: r, t: t, m: m }
    }
    pub fn to_column_vector(&self) -> OVector<f64, U6> {
        OVector::<f64, U6>::new(
            self.w[0], self.w[1], self.w[2], self.v[0], self.v[1], self.v[2],
        )
    }
    pub fn from_column_vector(col: &OVector<f64, U6>) -> Self {
        let w = OVector::<f64, U3>::new(col[0], col[1], col[2]);
        let v = OVector::<f64, U3>::new(col[3], col[4], col[5]);
        LieVectorSE3 { w: w, v: v }
    }

    pub fn hat(&self) -> OMatrix<f64, U4, U4> {
        hat4(&self)
    }

    pub fn from_hat(m: &OMatrix<f64, U4, U4>) -> Self {
        vee4(m)
    }
}

// SO3 hat
fn hat3(w: &OVector<f64, U3>) -> OMatrix<f64, U3, U3> {
    OMatrix::<f64, U3, U3>::new(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)
}

// SO3 vee
fn vee3(m: &OMatrix<f64, U3, U3>) -> OVector<f64, U3> {
    OVector::<f64, U3>::new(m.m32, m.m13, m.m21)
}

// SE3 hat
fn hat4(vec6: &LieVectorSE3) -> OMatrix<f64, U4, U4> {
    let mut m = OMatrix::<f64, U4, U4>::zeros();
    let top_left = hat3(&vec6.w);
    let top_right = &vec6.v;
    m.index_mut((0..3, 0..3)).copy_from(&top_left);
    m.index_mut((0..3, 3)).copy_from(top_right);
    m
}

// SE3 vee
fn vee4(m: &OMatrix<f64, U4, U4>) -> LieVectorSE3 {
    let mut top_left = OMatrix::<f64, U3, U3>::zeros();
    let mut top_right = OVector::<f64, U3>::zeros();
    top_left.copy_from(&m.slice((0, 0), (3, 3)));
    top_right.copy_from(&m.slice((0, 3), (3, 1)));
    let w = vee3(&top_left);
    LieVectorSE3 { w: w, v: top_right }
}

fn exp3(w: &OVector<f64, U3>) -> OMatrix<f64, U3, U3> {
    let theta = (w.dot(w)).sqrt();
    let w_hat = hat3(w);
    if theta < SMALL_FLOAT {
        let a = 1.0 - theta.powi(2) / 6.0 + theta.powi(4) / 120.0;
        let b = 0.5 - theta.powi(2) / 24.0 + theta.powi(4) / 720.0;
        OMatrix::<f64, U3, U3>::identity() + a * &w_hat + b * w_hat.pow(2)
    } else {
        OMatrix::<f64, U3, U3>::identity()
            + (theta.sin() / theta) * &w_hat
            + (1.0 - theta.cos()) / theta.powi(2) * w_hat.pow(2)
    }
}

fn log3(r: &OMatrix<f64, U3, U3>) -> Option<OVector<f64, U3>> {
    let theta = ((r.trace() - 1.0) / 2.0).acos();
    let delta_r = r - r.transpose();

    if theta.abs() < SMALL_FLOAT {
        let a = (1.0 - theta.powi(2) / 6.0 + theta.powi(4) / 120.0) / 2.0;
        let log_r = a * delta_r;
        let w = vee3(&log_r);
        Some(w)
    } else if (theta - PI).abs() < SMALL_FLOAT {
        // θ = π
        None
    } else {
        let a = theta / 2.0 / theta.sin();
        let log_r = a * delta_r;
        let w = vee3(&log_r);
        Some(w)
    }
}
