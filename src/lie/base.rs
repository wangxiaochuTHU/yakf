/// Generators of SE(3) are used as below:
///
/// G1 =    |0  0   0   0|
///         |0  0   -1  0|
///         |0  1   0   0|
///         |0  0   0   0|
///
/// G2 =    |0  0   1   0|
///         |0  0   0   0|
///         |-1 0   0   0|
///         |0  0   0   0|
///
/// G3 =    |0  -1  0   0|
///         |1  0   0   0|
///         |0  0   0   0|
///         |0  0   0   0|
///
/// G4 =    |0  0   0   1|
///         |0  0   0   0|
///         |0  0   0   0|
///         |0  0   0   0|
///
/// G5 =    |0  0   0   0|
///         |0  0   0   1|
///         |0  0   0   0|
///         |0  0   0   0|
///
/// G6 =    |0  0   0   0|
///         |0  0   0   0|
///         |0  0   0   1|
///         |0  0   0   0|
///
///  se(3) vector follows the order: [rotation, translation]    
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

const SMALL_FLOAT: f64 = 1e-6;

pub type LieAlgebraSE3 = OMatrix<f64, U4, U4>;

/// TODO: retain a best way to store the group.
/// This struct current is in redundancy.
/// both `(r,t)` and `m`, can determine the group.
#[derive(Debug, Clone)]
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
    pub fn to_vec6(&self) -> LieVectorSE3 {
        let (theta, w) = log3(&self.r);

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
        LieVectorSE3 { w: w, v: v }
    }

    pub fn from_r_t(r: OMatrix<f64, U3, U3>, t: OVector<f64, U3>) -> LieGroupSE3 {
        let mut m: OMatrix<f64, U4, U4> = OMatrix::<f64, U4, U4>::zeros();
        m.index_mut((0..3, 0..3)).copy_from(&r);
        m.index_mut((0..3, 3)).copy_from(&t);
        m[(3, 3)] = 1.0;
        LieGroupSE3 { r: r, t: t, m: m }
    }
    pub fn from_m(m: OMatrix<f64, U4, U4>) -> LieGroupSE3 {
        let mut r: OMatrix<f64, U3, U3> = OMatrix::<f64, U3, U3>::zeros();
        let mut t: OVector<f64, U3> = OVector::<f64, U3>::zeros();

        r.copy_from(&m.index((0..3, 0..3)));
        t.copy_from(&m.index((0..3, 3)));

        LieGroupSE3 { r: r, t: t, m: m }
    }

    pub fn inverse(&self) -> Self {
        let r2 = self.r.transpose();
        let t2 = -r2 * &self.t;
        let mut m = OMatrix::<f64, U4, U4>::zeros();
        m.index_mut((0..3, 0..3)).copy_from(&r2);
        m.index_mut((0..3, 3)).copy_from(&t2);
        m[(3, 3)] = 1.0;
        Self { r: r2, t: t2, m: m }
    }
    pub fn increment_by_left_delta(&mut self, delta: Self) {
        self.m = delta.m * &self.m;
        (self.r, self.t) = get_r_t_from_se3m(&self.m);
    }

    /// calculate the Adjoint matrix, with the definition of vector column form as [rotation_vec, translation_vec]
    /// [ R      0
    ///  [t]xR   R  ]
    ///
    pub fn adjoint_matrix(&self) -> OMatrix<f64, U6, U6> {
        let mut adj = OMatrix::<f64, U6, U6>::zeros();
        adj.index_mut((0..3, 0..3)).copy_from(&self.r);
        let bottom_left_block = hat3(&self.t) * &self.r;
        adj.index_mut((3..6, 0..3)).copy_from(&bottom_left_block);

        adj.index_mut((3..6, 3..6)).copy_from(&self.r);

        adj
    }

    ///  re-express vector v , from self's local frame to global frame
    pub fn adjoint_action(&self, v: &LieAlgebraSE3) -> LieAlgebraSE3 {
        let group_inv = self.inverse();

        self.m * v * group_inv.m
    }

    /// re-express point p, from self's local frame to global frame
    pub fn action_on_point(&self, p: &OVector<f64, U3>) -> OVector<f64, U3> {
        self.t + &self.r * p
    }

    /// get the group from a lgebra-formed matrix
    pub fn from_algebra(m: &LieAlgebraSE3) -> Self {
        let m = m.exp();
        let (r, t) = get_r_t_from_se3m(&m);
        LieGroupSE3 { r: r, t: t, m: m }
    }

    /// transform the group to lgebra-formed matrix
    pub fn to_algebra(&self) -> LieAlgebraSE3 {
        let vec6 = self.to_vec6();
        hat4(&vec6)
    }

    /// calculate the delta group, that starts from `self` and ends up at `end`
    pub fn delta_to_target(&self, end: &Self) -> Self {
        let start_inv = self.inverse();
        let m = end.m * start_inv.m;
        let (r, t) = get_r_t_from_se3m(&m);
        LieGroupSE3 { r: r, t: t, m: m }
    }

    /// equally interpolation between `start` and `end`.  start----[interpolation]----end
    /// nums_inter is the number of interpolation objects.
    pub fn interpolation(&self, end: &Self, nums_inter: usize) -> Vec<Self> {
        let delta = self.delta_to_target(end);
        let delta_algebra = delta.to_algebra();
        let d = 1.0 / (nums_inter + 1) as f64;
        let mut intergroups: Vec<Self> = Vec::new();
        for i in 0..nums_inter {
            let a = i as f64 * d;
            let d_delta_algebra = a * delta_algebra;
            let d_delta = d_delta_algebra.exp();
            let m = d_delta * &self.m;
            let (r, t) = get_r_t_from_se3m(&m);
            intergroups.push(LieGroupSE3 { r: r, t: t, m: m })
        }
        intergroups
    }
}

#[derive(Debug, Clone)]
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
        m.index_mut((0..3, 0..3)).copy_from(&w_hat);
        m.index_mut((0..3, 3)).copy_from(&self.v);

        let m = m.exp();

        LieGroupSE3 { r: r, t: t, m: m }
    }
    pub fn to_vec6(&self) -> OVector<f64, U6> {
        OVector::<f64, U6>::new(
            self.w[0], self.w[1], self.w[2], self.v[0], self.v[1], self.v[2],
        )
    }
    pub fn from_vec6(col: &OVector<f64, U6>) -> Self {
        let w = OVector::<f64, U3>::new(col[0], col[1], col[2]);
        let v = OVector::<f64, U3>::new(col[3], col[4], col[5]);
        LieVectorSE3 { w: w, v: v }
    }

    pub fn to_algebra(&self) -> LieAlgebraSE3 {
        hat4(&self)
    }

    pub fn from_algebra(m: &LieAlgebraSE3) -> Self {
        vee4(m)
    }
}

/// SO3 hat , vec3 --> algebra R{3×3}
fn hat3(w: &OVector<f64, U3>) -> OMatrix<f64, U3, U3> {
    OMatrix::<f64, U3, U3>::new(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)
}

/// SO3 vee,  algebra R{3×3} --> vec3
fn vee3(m: &OMatrix<f64, U3, U3>) -> OVector<f64, U3> {
    OVector::<f64, U3>::new(m.m32, m.m13, m.m21)
}

/// SE3 hat,  vec6  --> algebra R{4×4}
fn hat4(vec6: &LieVectorSE3) -> LieAlgebraSE3 {
    let mut m = OMatrix::<f64, U4, U4>::zeros();
    let top_left = hat3(&vec6.w);
    let top_right = &vec6.v;
    m.index_mut((0..3, 0..3)).copy_from(&top_left);
    m.index_mut((0..3, 3)).copy_from(top_right);
    m
}

/// pull blocks `r` and `t` from the matrix-formed Lie group
fn get_r_t_from_se3m(m: &OMatrix<f64, U4, U4>) -> (OMatrix<f64, U3, U3>, OVector<f64, U3>) {
    let mut r = OMatrix::<f64, U3, U3>::zeros();
    let mut t = OVector::<f64, U3>::zeros();
    r.copy_from(&m.slice((0, 0), (3, 3)));
    t.copy_from(&m.slice((0, 3), (3, 1)));
    (r, t)
}

/// SE3 vee,  algebra R{4×4} --> vec6
fn vee4(m: &LieAlgebraSE3) -> LieVectorSE3 {
    let mut top_left = OMatrix::<f64, U3, U3>::zeros();
    let mut top_right = OVector::<f64, U3>::zeros();
    top_left.copy_from(&m.slice((0, 0), (3, 3)));
    top_right.copy_from(&m.slice((0, 3), (3, 1)));
    let w = vee3(&top_left);
    LieVectorSE3 { w: w, v: top_right }
}

/// vec3 --> R{3×3}
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

/// R{3×3} --> vec3
/// this part takes [https://github.com/petercorke/spatialmath-matlab/blob/master/trlog.m] as a reference
fn log3(r: &OMatrix<f64, U3, U3>) -> (f64, OVector<f64, U3>) {
    let trace_r = r.trace();

    if (trace_r - 3.0).abs() < SMALL_FLOAT {
        // θ = 0
        let theta = 0.0;
        let w = OVector::<f64, U3>::zeros();
        (theta, w)
    } else if (trace_r + 1.0).abs() < SMALL_FLOAT {
        // θ = π
        let diag = r.diagonal();

        let (k, mx) = diag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let col = r.column(k) + OMatrix::<f64, U3, U3>::identity().column(k);
        let theta = PI;
        let w = col / (2.0 * (1.0 + mx)).sqrt();
        (theta, w)
    } else {
        // general case
        let theta = ((trace_r - 1.0) / 2.0).acos();
        let d_r = r - &r.transpose();
        if theta.abs() < SMALL_FLOAT {
            let a = (1.0 - theta.powi(2) / 6.0 + theta.powi(4) / 120.0) / 2.0;
            let log_r = a * d_r;
            let w = vee3(&log_r);
            (theta, w)
        } else {
            let a = theta / 2.0 / theta.sin();
            let log_r = a * d_r;
            let w = vee3(&log_r);
            (theta, w)
        }
    }
}
