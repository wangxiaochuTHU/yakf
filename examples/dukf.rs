/// import yakf crate
extern crate yakf;
/// import State trait, DUKF filter struct, and MSSS sampling method struct.
use yakf::kf::{DMinimalSkewSimplexSampling as DMSSS, DState, DUKF};
/// import Re-exports of hifitime (for time) and nalgebra (for matrix)
use yakf::{
    linalg,
    time::{Duration, Epoch, Unit},
};
fn main() {
    use crate::linalg::{DMatrix, DVector};
    use rand::prelude::*;

    #[derive(Debug)]
    /// define a custom struct to be the state. e.g., BikeState, has a 2-D vector x (x[0]: position, x[1]: velocity) and a timestamped time t.
    pub struct BikeState {
        pub x: DVector<f64>,
        pub t: Epoch,
    }

    /// for example, you can define your own methods.
    impl BikeState {
        pub fn new(state: DVector<f64>, epoch: Epoch) -> Self {
            BikeState { x: state, t: epoch }
        }
        pub fn zeros() -> Self {
            Self {
                x: DVector::<f64>::zeros(2),
                t: Epoch::from_gregorian_tai(2022, 5, 10, 0, 0, 0, 0),
            }
        }
    }

    /// you **MUST** implement State<T,U> for your custom state struct.
    ///
    impl DState for BikeState {
        fn state(&self) -> &DVector<f64> {
            &self.x
        }
        fn set_state(&mut self, state: DVector<f64>) {
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
    let dynamics = |x: &DVector<f64>, _ext: &DVector<f64>, dt: Duration| {
        DVector::<f64>::from_row_slice(&[x[0] + x[1] * dt.in_seconds(), x[1]])
    };

    // you SHOULD ALSO provide a function for UKF yielding measurements based on given state.
    //
    // for example, assume the measuring has a 2-D measurement.
    let measure_model = |x: &DVector<f64>| DVector::<f64>::from_row_slice(&[x[0], x[1]]);

    // you SHOULD ALSO specify a sampling method for UKF.
    // for example, you can specify a MSSS method

    let n = 2_usize;
    let n2 = 4_usize;
    let m = 2_usize;
    let samling_method = DMSSS::build(0.6, n).unwrap();

    // // or you can specify a SDS method as an alternative.
    // type _T2 = Const<5>;

    // let _samling_method = SDS::<U2, _T2>::build(1e-3, None, None).unwrap();

    // finally, build the UKF.
    let mut dukf = DUKF::<BikeState>::build(
        n,
        n2,
        m,
        Box::new(dynamics),
        Box::new(measure_model),
        Box::new(samling_method),
        BikeState::zeros(),
        DMatrix::<f64>::from_diagonal(&DVector::<f64>::from_row_slice(&[10.0, 10.0])),
        DMatrix::<f64>::from_diagonal(&DVector::<f64>::from_row_slice(&[1.0, 1.0])),
        DMatrix::<f64>::from_diagonal(&DVector::<f64>::from_row_slice(&[1.0, 0.001])),
    )
    .unwrap();

    // you can then use ukf to estimate the state vector.

    let mut rng = rand::thread_rng();
    let mut add_noisies = |mut y: DVector<f64>| {
        y[0] += rng.gen_range(-3.0..3.0);
        y[1] += rng.gen_range(-0.1..0.1);
        y
    };
    let s = DVector::<f64>::from_row_slice(&[-5.0, 1.0]);
    let t = Epoch::from_gpst_nanoseconds(0);
    let mut bike_actual = BikeState::new(s, t);

    println!(
        "bike actual = {:?}, ukf estimate = {:?}",
        &bike_actual,
        &dukf.current_estimate()
    );

    // you can set an arbitary time base for ukf.
    // a timing system would help in aligning data.
    let ukf_base_epoch = dukf.current_estimate().epoch();
    let mut actual_normed_noise: Vec<f64> = Vec::new();
    let mut estimate_normed_error: Vec<f64> = Vec::new();
    let nums_measure = 500_usize;

    for i in 0..nums_measure {
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
        let _ = bike_actual.propagate(&dynamics, dt, DVector::<f64>::from_row_slice(&[0.0]));

        // use measuring model to simulate a measurement, and add some noises on it.
        let mut meas = measure_model(&bike_actual.state());
        meas = add_noisies(meas);

        // every time the measurement is ready, ukf is trigger to update.
        dukf.feed_and_update(
            meas.clone(),
            m_epoch,
            DVector::<f64>::from_row_slice(&[0.0]),
        );
        if i > nums_measure / 3 {
            actual_normed_noise.push((&meas - bike_actual.state()).norm());
            estimate_normed_error
                .push((dukf.current_estimate().state() - bike_actual.state()).norm());
        }

        println!(
            "bike actual = {:?}, meas = {:.3?}, ukf estimate = {:.3?}",
            &bike_actual.state().as_slice(),
            meas.as_slice(),
            &dukf.current_estimate().state().as_slice(),
        );
    }

    let nums = actual_normed_noise.len();
    let noise_metric: f64 = actual_normed_noise
        .into_iter()
        .fold(0.0, |acc, x| acc + x / nums as f64);
    let error_metric: f64 = estimate_normed_error
        .into_iter()
        .fold(0.0, |acc, x| acc + x / nums as f64);
    println!(
        "noise_metric = {:?}, error_metric = {:?}",
        noise_metric, error_metric
    );
    assert!(error_metric < noise_metric);
}
