use crate::linalg::{Const, OMatrix, OVector, U2, U3};
use sosekf::SOEKF;
use yakf::lie::so3::{Alg3, Grp3, One2OneMapSO3, Vec3, SO3};
/// import Re-exports of hifitime (for time) and nalgebra (for matrix)
use yakf::{
    linalg,
    time::{Duration, Epoch, Unit},
};
fn main() {
    use core::f64::consts::PI;
    use rand::Rng;

    let state = SO3::from_vec(Vec3::new(0.3, 0.6, -0.9));
    let pmatrix = OMatrix::<f64, U3, U3>::from_diagonal_element(1e-2);
    let qmatrix = OMatrix::<f64, U3, U3>::from_diagonal_element(1e-2);
    let nmatrix = OMatrix::<f64, Const<12>, Const<12>>::from_diagonal_element(1e-4);

    let mut soekf = SOEKF::build(state, pmatrix, qmatrix, nmatrix);

    let nums = 2000;
    let bk = [
        OVector::<f64, U3>::new(10.0, 20.0, 10.0),
        OVector::<f64, U3>::new(-50.0, 30.0, 10.0),
        OVector::<f64, U3>::new(50.0, 20.0, -10.0),
        OVector::<f64, U3>::new(-20.0, -10.0, -5.0),
    ];

    let true_pose_and_u = |dur: Duration| {
        let omega = 0.3;

        let theta = omega * dur.in_seconds();

        let e1 = OVector::<f64, U3>::new(theta.cos(), theta.sin(), 0.0);
        let e2 = OVector::<f64, U3>::new((theta + PI / 2.0).cos(), (theta + PI / 2.0).sin(), 0.0);
        let e3 = OVector::<f64, U3>::new(0.0, 0.0, 1.0);
        let r = OMatrix::<f64, U3, U3>::from_columns(&[e1, e2, e3]);
        let x = SO3::from_grp(r);

        let u = OVector::<f64, U3>::new(0.0, 0.0, omega);

        (x, u)
    };
    pub fn measure(x_predict: &SO3, bk: &[Vec3; 4]) -> OVector<f64, Const<12>> {
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
        println!("estimate error = {}", error);
    }

    assert!(error < 1e-2);
}

pub mod sosekf {
    use yakf::time::{Duration, Epoch};

    use yakf::errors::YakfError;
    use yakf::lie::so3::{hat, jac_r, Alg3, Grp3, One2OneMapSO3, Vec3, SO3};

    use yakf::linalg::allocator::Allocator;
    use yakf::linalg::{Const, DefaultAllocator, DimName, OMatrix, OVector, U3, U4, U6};
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
