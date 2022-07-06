use seekf::SEEKF;
use yakf::lie::se3::combine;
use yakf::lie::se3::{Alg6, Grp6, One2OneMapSE, Vec6, SE3};
use yakf::lie::so3::{Alg3, Grp3, One2OneMap, Vec3, SO3};

use yakf::linalg::{Const, OMatrix, OVector, U2, U3, U4, U6};
/// import Re-exports of hifitime (for time) and nalgebra (for matrix)
use yakf::{
    linalg,
    time::{Duration, Epoch, Unit},
};

fn main() {
    use core::f64::consts::PI;
    use rand::Rng;

    let state = SE3::from_vec(Vec6::new(0.3, 0.6, -0.9, 0.4, 0.3, 0.2));
    let pmatrix = OMatrix::<f64, U6, U6>::from_diagonal_element(1e2);
    let qmatrix = OMatrix::<f64, U6, U6>::from_diagonal_element(1e-2);
    let nmatrix = OMatrix::<f64, Const<12>, Const<12>>::from_diagonal_element(1e-4);

    let mut soekf = SEEKF::build(state, pmatrix, qmatrix, nmatrix);

    let nums = 200;
    let bk = [
        OVector::<f64, U3>::new(10.0, 20.0, 10.0),
        OVector::<f64, U3>::new(-50.0, 30.0, 10.0),
        OVector::<f64, U3>::new(50.0, 20.0, -10.0),
        OVector::<f64, U3>::new(-20.0, -10.0, -5.0),
    ];

    let true_pose_and_u = |dur: Duration| {
        let omega = 0.3;

        let theta = omega * dur.in_seconds();
        let ro = 10.0;

        let e1 = OVector::<f64, U3>::new(theta.cos(), theta.sin(), 0.0);
        let e2 = OVector::<f64, U3>::new((theta + PI / 2.0).cos(), (theta + PI / 2.0).sin(), 0.0);
        let e3 = OVector::<f64, U3>::new(0.0, 0.0, 1.0);
        let r = OMatrix::<f64, U3, U3>::from_columns(&[e1, e2, e3]);

        let t = ro * OVector::<f64, U3>::new(theta.cos(), theta.sin(), 0.0);
        let x = SE3::from_r_t(r, t);

        let v = OVector::<f64, U3>::new(0.0, ro * omega, 0.0);
        let w = OVector::<f64, U3>::new(0.0, 0.0, omega);
        let u = combine(v, w);

        (x, u)
    };
    pub fn measure(x_predict: &SE3, bk: &[Vec3; 4]) -> OVector<f64, Const<12>> {
        let mut ob = OVector::<f64, Const<12>>::zeros();
        let x_inv = x_predict.inverse();
        for i in 0..4 {
            let block = x_inv.act_v(bk[i]);
            ob.index_mut((i * 3..i * 3 + 3, 0..1)).copy_from(&block);
        }
        ob
    }

    let mut rng = rand::thread_rng();
    let mut add_noisies = |mut y: OVector<f64, Const<12>>| {
        for i in 0..12 {
            y[i] += rng.gen_range(-1e-2..1e-2);
        }
        y
    };

    let mut dur = Duration::from_f64(0.0, Unit::Second);
    let dt = Duration::from_f64(0.1, Unit::Second);
    let mut error: f64 = 1.0;
    for i in 0..nums {
        dur = dur + dt;

        let (true_state, true_u_b) = true_pose_and_u(dur);
        let meas_ob = {
            let mut true_ob = measure(&true_state, &bk);
            let meas_ob = add_noisies(true_ob);
            meas_ob
        };

        // every time the measurement is ready, ekf is trigger to update.
        soekf.feed_and_update(meas_ob, dt, true_u_b, &bk).unwrap();

        let estimate_state = soekf.state;
        let error_state = estimate_state.minus_r(true_state);
        error = error_state.norm();
        println!("error = {}", error);
    }

    assert!(error < 1e-2);
}

pub mod seekf {
    use yakf::time::{Duration, Epoch};

    use yakf::errors::YakfError;
    use yakf::lie::se3::{combine, decombine, hat, jac_r, Alg6, Grp6, One2OneMapSE, Vec6, SE3};
    use yakf::lie::so3::{hat as hat3, Vec3};

    use se3_kf::linalg::allocator::Allocator;
    use se3_kf::linalg::{Const, DefaultAllocator, DimName, OMatrix, OVector, U3, U4, U6};
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
        }
    }
}
