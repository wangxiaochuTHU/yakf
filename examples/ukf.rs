/// import yakf crate
extern crate yakf;
/// import State trait, UKF filter struct, and MSSS sampling method struct
use yakf::kf::{MinimalSkewSimplexSampling as MSSS, State, UKF};

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
    // you SHOULD provide a function `dynamics` for UKF propagating the state.
    //
    // for example,
    let dynamics = |x: &OVector<f64, U2>, _ext: &OVector<f64, Const<1>>, dt: Duration| {
        OVector::<f64, U2>::new(x[0] + x[1] * dt.in_seconds(), x[1])
    };

    // you SHOULD ALSO provide a function for UKF yielding measurements based on given state.
    //
    // for example, assume the measuring has a 2-D measurement.
    let measure_model =
        |x: &OVector<f64, U2>| OVector::<f64, U2>::new(5.0 * x[0] * x[0] * x[1], x[1] * x[0] * 2.0);

    // you SHOULD ALSO specify a sampling method for UKF.
    // currently, only MSSS method is done. MSSS has only one parameter `w0` to tune.
    // w0 belongs to [0.0 , 1.0].  this example takes w0 = 0.6.
    let samling_method = MSSS::build(0.6).unwrap();

    // finally, build the UKF.
    let mut ukf = UKF::<U2, Const<4>, U2, Const<1>, BikeState>::build(
        Box::new(dynamics),
        Box::new(measure_model),
        Box::new(samling_method),
        BikeState::zeros(),
        OMatrix::<f64, U2, U2>::from_diagonal_element(1.0),
        OMatrix::<f64, U2, U2>::from_diagonal_element(0.01),
        OMatrix::<f64, U2, U2>::from_diagonal(&OVector::<f64, U2>::new(1.0, 0.01)),
    );

    // you can then use ukf to estimate the state vector.

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

    // you can set an arbitary time base for ukf.
    // a timing system would help in aligning data.
    let ukf_base_epoch = ukf.current_estimate().epoch();

    for _i in 0..1000 {
        let dt = Duration::from_f64(1.0, Unit::Second);
        let m_epoch = ukf_base_epoch + dt;

        /*
        Remark 1. Note that the actual dynamics doesn't need to be exactly the same with that used by ukf.
                Actually, the dynamics used by ukf is only a Model abstracted from the actual one.
                But in this example, assume they are the same. Case is the same for measuring model.

        Remark 2. For the same reason, the delta_t used by actual dynamics is not neccesarily the same
                with dt (the one used by ukf estimation) and, actually, delta_t should be much smaller than dt
                in real world. However, for simplity, this example just let them be the same, i.e. delta_t = dt.
        */
        let _ = bike_actual.propagate(&dynamics, dt, OVector::<f64, Const<1>>::zeros());

        // use measuring model to simulate a measurement, and add some noises on it.
        let mut meas = measure_model(&bike_actual.state());
        meas = add_noisies(meas);

        // every time the measurement is ready, ukf is trigger to update.
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
