/// import yakf crate
extern crate yakf;
/// import State trait, UKF filter struct, and MSSS sampling method struct
use yakf::lie::base::{hat3, LieAlgebraSE3, LieGroupSE3, LieVectorSE3};
use yakf::sfilters::sekf::{ESStates, ESEKF};

use core::f64::consts::PI;
use linalg::{Const, OMatrix, OVector, U2, U3, U4, U6};
use rand::prelude::*;
/// import Re-exports of hifitime (for time) and nalgebra (for matrix)
use yakf::{
    linalg,
    time::{Duration, Epoch, Unit},
};
const beacons: [OVector<f64, U3>; 4] = [
    OVector::<f64, U3>::new(50.0, 50.0, 10.0),
    OVector::<f64, U3>::new(-50.0, -50.0, 10.0),
    OVector::<f64, U3>::new(50.0, 50.0, -10.0),
    OVector::<f64, U3>::new(-50.0, -50.0, -10.0),
];

fn main() {
    /// define a custom struct to be the state. e.g., BikeState, has a 2-D vector x (x[0]: position, x[1]: velocity) and a timestamped time t.
    pub struct PlanePose {
        pub x: LieGroupSE3,
        pub t: Epoch,
    }

    /// for example, you can define your own methods.
    impl PlanePose {
        pub fn new(state: LieGroupSE3, epoch: Epoch) -> Self {
            PlanePose { x: state, t: epoch }
        }
        pub fn zeros() -> Self {
            Self {
                x: LieGroupSE3::zeros(),
                t: Epoch::from_gregorian_tai(2022, 5, 10, 0, 0, 0, 0),
            }
        }
    }

    impl ESStates for PlanePose {
        fn state(&self) -> &LieGroupSE3 {
            &self.x
        }
        fn set_state(&mut self, state: LieGroupSE3) {
            self.x = state;
        }

        fn epoch(&self) -> Epoch {
            self.t
        }
        fn set_epoch(&mut self, epoch: Epoch) {
            self.t = epoch;
        }
    }
    // let dynamics = |x: &OVector<f64, U2>, _ext: &OVector<f64, Const<1>>, dt: Duration| {
    //     OVector::<f64, U2>::new(x[0] + x[1] * dt.in_seconds(), x[1])
    // };
    // let beacons = [
    //     OVector::<f64, U3>::new(50.0, 50.0, 10.0),
    //     OVector::<f64, U3>::new(-50.0, -50.0, 10.0),
    //     OVector::<f64, U3>::new(50.0, 50.0, -10.0),
    //     OVector::<f64, U3>::new(-50.0, -50.0, -10.0),
    // ];

    let f = |x: &LieGroupSE3, u: &LieVectorSE3, dt: Duration| OMatrix::<f64, U6, U6>::identity();
    let g =
        |x: &LieGroupSE3, u: &LieVectorSE3, dt: Duration| x.adjoint_matrix().try_inverse().unwrap();
    let ob = |x: &LieGroupSE3| {
        let mut meas = OVector::<f64, Const<12>>::zeros();
        let x_inv = x.inverse();
        for k in 0..beacons.len() {
            let p = beacons[k];
            meas.index_mut((3 * k..3 * k + 3, 0))
                .copy_from(&x_inv.action_on_point(&p))
        }
        meas
    };

    let h = |x: &LieGroupSE3| {
        let mut m = OMatrix::<f64, Const<12>, U6>::zeros();

        // for k in 0..beacons.len() {
        //     let p = beacons[k];
        //     let mut mm = OMatrix::<f64, U3, U6>::zeros();
        //     let p_hat = hat3(&p);
        //     let r = x.r;
        //     let t = x.t;
        //     let xx = r.transpose() * hat3(&OVector::<f64, U3>::from_element(1.0)) * (p - t);
        //     mm.index_mut((0..3, 0..3)).copy_from(&());

        //     mm.index_mut((0..3, 3..6))
        //         .copy_from(&(-OMatrix::<f64, U3, U3>::identity()));
        // }

        let adx = x.adjoint_matrix().try_inverse().unwrap();

        for k in 0..beacons.len() {
            let p = beacons[k];
            let mut mm = OMatrix::<f64, U3, U6>::zeros();
            let p_hat = hat3(&p);
            let r = x.r;
            let t = x.t;

            mm.index_mut((0..3, 0..3)).copy_from(&(r * p_hat));

            mm.index_mut((0..3, 3..6)).copy_from(&(-r));

            m.index_mut((3 * k..3 * k + 3, 0..6)).copy_from(&(mm * adx));
        }
        m
    };

    let t_base = Epoch::now().unwrap();
    let mut init_guess = PlanePose::zeros();
    init_guess.set_epoch(t_base);

    let mut eskf = ESEKF::build(
        Box::new(f),
        Box::new(g),
        Box::new(h),
        Box::new(ob),
        init_guess,
        OMatrix::<f64, U6, U6>::from_diagonal_element(10.0),
        OMatrix::<f64, U6, U6>::from_diagonal_element(0.1),
        OMatrix::<f64, Const<12>, Const<12>>::from_diagonal_element(0.1),
    );

    let mut rng = rand::thread_rng();
    let mut add_noisies = |mut y: OVector<f64, Const<12>>| {
        for i in 0..12 {
            y[i] += rng.gen_range(-0.1..0.1);
        }
        y
    };

    let true_pose_and_u = |t: Epoch| {
        let omega = 0.3;
        let dt = t - t_base;
        let theta = omega * dt.in_seconds();
        let ro = 100.0;
        let t = OVector::<f64, U3>::new(ro * theta.cos(), ro * theta.sin(), 0.0);
        let e1 = OVector::<f64, U3>::new(theta.cos(), theta.sin(), 0.0);
        let e2 = OVector::<f64, U3>::new((theta + PI / 2.0).cos(), (theta + PI / 2.0).sin(), 0.0);
        let e3 = OVector::<f64, U3>::new(0.0, 0.0, 1.0);
        let r_inv = OMatrix::<f64, U3, U3>::from_columns(&[e1, e2, e3]);
        let r = r_inv.transpose();
        let x = LieGroupSE3::from_r_t(r, t);
        let w = OVector::<f64, U3>::new(0.0, 0.0, omega);
        let v_b = OVector::<f64, U3>::new(0.0, ro * omega, 0.0);
        let u_b = LieVectorSE3 { w: w, v: v_b };

        let u_i = LieVectorSE3::from_vec6(&(x.adjoint_matrix() * u_b.to_vec6()));
        (x, u_i, u_b)
    };

    let nums_measure = 500_usize;
    let dt = Duration::from_f64(10.0, Unit::Millisecond);
    for i in 0..nums_measure {
        let m_epoch = t_base + dt * i as f64;

        let (true_pose, true_u_i, true_u_b) = true_pose_and_u(m_epoch);
        let meas_ob = {
            let mut true_ob = ob(&true_pose);
            let meas_ob = add_noisies(true_ob);
            meas_ob
        };

        // every time the measurement is ready, ekf is trigger to update.
        eskf.feed_and_update(meas_ob, m_epoch, true_u_i).unwrap();

        // if i > nums_measure / 3 {
        //     actual_normed_noise.push((&meas - bike_actual.state()).norm());
        //     estimate_normed_error
        //         .push((dekf.current_estimate().state() - bike_actual.state()).norm());
        // }

        let plane_estimate = eskf.stamp_state.state();
        let mut error = true_pose.inverse();
        error.increment_by_left_delta(plane_estimate.clone());

        println!(
            "plane_estimate - plane_true = {:?}",
            error.to_vec6().to_vec6().norm(),
        );
        // println!("plane_estimate.r = {:?}", eskf.stamp_state.state().r,);
        // println!("plane_estimate.t = {:?}", eskf.stamp_state.state().t,);
    }

    // println!(
    //     "bike actual = {:?}, ukf estimate = {:?}",
    //     &bike_actual,
    //     &dekf.current_estimate()
    // );
}
