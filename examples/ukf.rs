/// import yakf crate
extern crate yakf;
/// import State trait, UKF filter struct, and MSSS sampling method struct
use yakf::kf::{sigma_points::MinimalSkewSimplexSampling as MSSS, state::State, ukf::UKF};

/// import Re-exports of hifitime (for time) and nalgebra (for matrix)
use yakf::{
    linalg,
    time::{Duration, Epoch, Unit},
};

fn main() {
    use crate::linalg::{Const, OMatrix, OVector, U2};
    use rand::prelude::*;

    #[derive(Debug)]
    /// define a custom struct to be the state. e.g., BikeState, has a 2-D vector x and a timestamped time t.
    pub struct BikeState {
        pub x: OVector<f64, U2>,
        pub t: Epoch,
    }

    /// for example, you can define your own methods.
    impl BikeState {
        pub fn new(state: OVector<f64, U2>, epoch: Epoch) -> Self {
            BikeState { x: state, t: epoch }
        }
        pub fn zeros() -> Self {
            Self {
                x: OVector::<f64, U2>::zeros(),
                t: Epoch::from_gregorian_tai(2022, 5, 10, 0, 0, 0, 0),
            }
        }
    }

    /// you **MUST** implement State<T,U> for your custom state struct.
    ///
    impl State<U2, Const<1>> for BikeState {
        fn state(&self) -> OVector<f64, U2> {
            self.x.clone()
        }
        fn set_state(&mut self, state: OVector<f64, U2>) {
            self.x = state;
        }

        fn epoch(&self) -> Epoch {
            self.t
        }
        fn set_epoch(&mut self, epoch: Epoch) {
            self.t = epoch;
        }
    }
    // you SHOULD provide a function `dynamics` for propagating the state.
    //
    // for example,
    let dynamics = |x: &OVector<f64, U2>, _ext: &OVector<f64, Const<1>>, dt: Duration| {
        OVector::<f64, U2>::new(x[0] + x[1] * dt.in_seconds(), x[1])
    };

    // you SHOULD ALSO provide a function for yielding measurements based on given state.
    //
    // for example, assume the measuring has a 2-D measurement.
    let measure_model =
        |x: &OVector<f64, U2>| OVector::<f64, U2>::new(5.0 * x[0] * x[0] * x[1], x[1] * x[0] * 2.0);

    // Finally, build the UKF
    let mut ukf = UKF::<U2, Const<4>, U2, Const<1>, BikeState>::build(
        Box::new(dynamics),
        Box::new(measure_model),
        Box::new(MSSS::build(0.6)),
        BikeState::zeros(),
        OMatrix::<f64, U2, U2>::from_diagonal_element(1.0),
        OMatrix::<f64, U2, U2>::from_diagonal_element(0.01),
        OMatrix::<f64, U2, U2>::from_diagonal(&OVector::<f64, U2>::new(1.0, 0.01)),
    );

    // You can then use ukf to estimate the state vector.

    let mut rng = rand::thread_rng();
    let mut add_noisies = |mut y: OVector<f64, U2>| {
        y[0] += rng.gen_range(-1.0..1.0);
        y[1] += rng.gen_range(-0.1..0.1);
        y
    };
    let s = OVector::<f64, U2>::new(-5.0, 1.0);
    let t = Epoch::now().unwrap();
    let mut bike_actual = BikeState::new(s, t);

    println!(
        "bike actual = {:?}, ukf estimate = {:?}",
        &bike_actual,
        &ukf.current_estimate()
    );

    let ukf_base_epoch = ukf.current_estimate().epoch();
    for _i in 0..1000 {
        let dt = Duration::from_f64(1.0, Unit::Second);
        let m_epoch = ukf_base_epoch + dt;
        let _ = bike_actual.propagate(&dynamics, dt, OVector::<f64, Const<1>>::zeros());
        let mut meas = measure_model(&bike_actual.state());
        meas = add_noisies(meas);
        ukf.feed_and_update(meas, m_epoch, OVector::<f64, Const<1>>::zeros());

        println!(
            "bike actual = {:?}, meas = {:.3?}, ukf estimate = {:.3?}",
            &bike_actual.state(),
            meas,
            &ukf.current_estimate().state(),
        );
    }
    let error = &ukf.current_estimate().state() - &bike_actual.state();
    assert!(error.norm() < 0.5);
}
