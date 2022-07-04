# yakf - Yet Another Kalman Filter
Yet Another Kalman Filter Implementation, as well as, 

Lie Theory (Lie group, algebra, vector) on SO(3) and SE(3).

`[no_std]` is supported by default.

# Current implementation status

## Filter Status
* UKF ✅ 
* EKF (Only dynamically-sized version) ✅ 
* ***(unfinished, non-generic) Kalman Filter on SO(3)*** ✅ refer to `yakf::lie::so3::sosekf::SOEKF`

## Sampling Method Status
* Minimal Skew Simplex Sampling (n+2) ✅
* Symmetrically-Distributed Sampling Method (2n+1) ✅

## Static V.S Dynamic Cases
* For ***statically***-sized state whose dimension is known in compile time, refer to `yakf::filters`
* For ***dynamically***-sized state whose dimension may vary in run time, refer to `yakf::dfilters`

## Lie Group Status
* SO(3) ✅
* SE(3) ✅


***NOTE that some functions havn't been thoroughly tested, so please let me know if there is any error.***

# Usage
Add this to your Cargo.toml:
```
[dependencies]
yakf = "0.1"
```

Example (statically-sized):
```
/// import yakf crate
extern crate yakf;
/// import State trait, UKF filter struct, and MSSS sampling method struct
use yakf::kf::{
    MinimalSkewSimplexSampling as MSSS, State, SymmetricallyDistributedSampling as SDS, UKF,
};

/// import Re-exports of hifitime (for time) and nalgebra (for matrix)
use yakf::{
    linalg,
    time::{Duration, Epoch, Unit},
};

fn main() {
    use crate::linalg::{Const, OMatrix, OVector, U2};
    use rand::prelude::*;

    #[derive(Debug)]
    /// define a custom struct to be the state. e.g., BikeState, has a 2-D vector x (x[0]: position, x[1]: velocity) and a timestamped time t.
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
        fn state(&self) -> &OVector<f64, U2> {
            &self.x
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
    let measure_model = |x: &OVector<f64, U2>| OVector::<f64, U2>::new(x[0], x[1]);

    // you SHOULD ALSO specify a sampling method for UKF.
    // for example, you can specify a MSSS method
    type T2 = Const<4>;
    let samling_method = MSSS::<U2, T2>::build(0.6).unwrap();

    // or you can specify a SDS method as an alternative.
    type _T2 = Const<5>;

    let _samling_method = SDS::<U2, _T2>::build(1e-3, None, None).unwrap();

    // finally, build the UKF.
    let mut ukf = UKF::<U2, T2, U2, Const<1>, BikeState>::build(
        Box::new(dynamics),
        Box::new(measure_model),
        Box::new(samling_method),
        BikeState::zeros(),
        OMatrix::<f64, U2, U2>::from_diagonal_element(10.0),
        OMatrix::<f64, U2, U2>::from_diagonal_element(1.0),
        OMatrix::<f64, U2, U2>::from_diagonal(&OVector::<f64, U2>::new(1.0, 0.001)),
    );

    // you can then use ukf to estimate the state vector.

    let mut rng = rand::thread_rng();
    let mut add_noisies = |mut y: OVector<f64, U2>| {
        y[0] += rng.gen_range(-3.0..3.0);
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
    let mut actual_normed_noise: Vec<f64> = Vec::new();
    let mut estimate_normed_error: Vec<f64> = Vec::new();
    let nums_measure = 500_usize;

    // you can set an arbitary time base for ukf.
    // a timing system would help in aligning data.
    let ukf_base_epoch = ukf.current_estimate().epoch();

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
        let _ = bike_actual.propagate(&dynamics, dt, OVector::<f64, Const<1>>::zeros());

        // use measuring model to simulate a measurement, and add some noises on it.
        let mut meas = measure_model(&bike_actual.state());
        meas = add_noisies(meas);

        // every time the measurement is ready, ukf is trigger to update.
        ukf.feed_and_update(meas, m_epoch, OVector::<f64, Const<1>>::zeros());
        if i > nums_measure / 3 {
            actual_normed_noise.push((&meas - bike_actual.state()).norm());
            estimate_normed_error
                .push((ukf.current_estimate().state() - bike_actual.state()).norm());
        }

        println!(
            "bike actual = {:?}, meas = {:.3?}, ukf estimate = {:.3?}",
            &bike_actual.state(),
            meas,
            &ukf.current_estimate().state(),
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




```

You may see the output as
```
.. .. ..
actual = [493.0, 1.0], meas = [493.281, 1.073], estimate = [492.553, 1.073]
actual = [494.0, 1.0], meas = [492.615, 0.941], estimate = [492.598, 0.941]
actual = [495.0, 1.0], meas = [496.849, 1.019], estimate = [495.710, 1.019]
noise_metric = 1.5346849337852513, error_metric = 1.2218914483371828
```