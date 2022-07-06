use yakf::lie::so3::{sosekf::SOEKF, Alg3, Grp3, One2OneMap, Vec3, SO3};

use crate::linalg::{Const, OMatrix, OVector, U2, U3};
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
