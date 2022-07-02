#[cfg(test)]
mod tests {
    use hifitime::Duration;

    use crate::filters::state::State;

    #[test]
    fn test_state() {
        use crate::filters::state;

        use crate::linalg::{Const, OVector, U2};
        extern crate libc_print;
        use crate::time::{Epoch, Unit};
        use libc_print::libc_println;

        #[derive(Debug)]
        pub struct BikeState {
            pub x: OVector<f64, U2>,
            pub t: Epoch,
        }
        impl state::State<Const<2>, Const<1>> for BikeState {
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

        impl BikeState {
            fn new(state: OVector<f64, U2>, epoch: Epoch) -> Self {
                BikeState { x: state, t: epoch }
            }
        }

        let s = OVector::<f64, U2>::new(-5.0, 1.0);
        let t = Epoch::now().unwrap();
        let mut bike = BikeState::new(s, t);
        let dynamics = |x: &OVector<f64, U2>, _ext: &OVector<f64, Const<1>>, dt: Duration| {
            OVector::<f64, U2>::new(x[0] + x[1] * dt.in_seconds(), x[1])
        };
        // libc_println!("bike state =  {:?}", bike);
        let _ = bike.propagate(
            &dynamics,
            Duration::from_f64(1.0, Unit::Second),
            OVector::<f64, Const<1>>::zeros(),
        );
        // libc_println!("bike state =  {:?}", bike);
    }

    /*
    /// This test relies on std::
    #[test]
    fn test_manually_expand_bases() {
        use std::collections::HashMap;
        let n = 7_usize;
        let mut u: HashMap<(usize, usize), Vec<String>> = HashMap::new();
        let x = "x".to_string();
        u.insert((0, 1), vec!["0".to_string()]);
        u.insert((1, 1), vec!["-w1".to_string()]);
        u.insert((2, 1), vec!["w2".to_string()]);

        let zero = |k: usize| {
            let mut v = Vec::new();
            for _ in 0..k {
                v.push("0".to_string());
            }
            v
        };
        let mut w: Vec<String> = Vec::new();
        let mut w_: Vec<String> = Vec::new();
        for i in 0..n + 2 {
            let x = format!("w{}", i);
            let x_ = format!("-w{}", i);
            w.push(x);
            w_.push(x_);
        }

        for j in 2..n + 1 {
            for i in 0..n + 2 {
                if i == 0 {
                    if let Some(s) = u.get(&(i, j - 1)) {
                        let mut v = s.clone();
                        v.push("0".to_string());
                        u.insert((i, j), v);
                    }
                } else if i == j + 1 {
                    let mut v = zero(j - 1);
                    v.push(w[j].clone());
                    u.insert((i, j), v);
                } else {
                    if let Some(s) = u.get(&(i, j - 1)) {
                        let mut v = s.clone();
                        v.push(w_[j].clone());
                        u.insert((i, j), v);
                    }
                }
            }
        }
        libc_println!("u_0n = {:#?}", u.get(&(0, n)).unwrap());
        libc_println!("u_1n = {:#?}", u.get(&(1, n)).unwrap());
        libc_println!("u_2n = {:#?}", u.get(&(2, n)).unwrap());
        libc_println!("u_3n = {:#?}", u.get(&(3, n)).unwrap());
        libc_println!("u_4n = {:#?}", u.get(&(4, n)).unwrap());
        libc_println!("u_5n = {:#?}", u.get(&(5, n)).unwrap());
        libc_println!("u_6n = {:#?}", u.get(&(6, n)).unwrap());
        libc_println!("u_7n = {:#?}", u.get(&(7, n)).unwrap());
        libc_println!("u_8n = {:#?}", u.get(&(8, n)).unwrap());
    }
    */
    #[test]
    fn test_sampling_weights_minimal_skew_simplex_sampling() {
        use crate::filters::sigma_points::MinimalSkewSimplexSampling;

        use crate::linalg::{Const, OVector, RealField, U3};
        use crate::time::Epoch;
        extern crate libc_print;
        use libc_print::libc_println;

        #[derive(Debug)]
        pub struct CarState<T>
        where
            T: RealField + Copy,
        {
            pub x: OVector<T, U3>,
            pub t: Epoch,
        }
        let sampling: MinimalSkewSimplexSampling<Const<6>, Const<8>> =
            MinimalSkewSimplexSampling::build(0.6).unwrap();

        // libc_println!("weights = {:?}", sampling.weights);
        let sum_w = sampling.weights.sum();
        // libc_println!("sum_w = {:?}", sum_w);
        assert!((sum_w - 1.0).abs() < 1e-11);

        let weighted_sum_u = sampling
            .weights
            .iter()
            .zip(sampling.u_bases.unwrap().column_iter())
            .fold(OVector::<f64, Const<6>>::zeros(), |acc, (w, u)| {
                acc + *w * u
            });
        // libc_println!("weighted_sum_u = {:#?}", weighted_sum_u);
        assert!(weighted_sum_u.norm() < 1e-11);
    }
    #[test]
    fn test_ukf_minimal_skew_simplex_sampling() {
        use crate::filters::sigma_points::MinimalSkewSimplexSampling;
        use crate::filters::state;
        use crate::filters::ukf::UKF;

        use crate::linalg::{Const, OMatrix, OVector, U2};
        use crate::time::{self, Epoch, Unit};
        use alloc::{boxed::Box, vec::Vec};
        use rand::prelude::*;
        extern crate libc_print;
        use libc_print::libc_println;

        #[derive(Debug)]
        pub struct BikeState {
            pub x: OVector<f64, U2>,
            pub t: Epoch,
        }
        impl state::State<Const<2>, Const<1>> for BikeState {
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

        impl BikeState {
            pub fn new(state: OVector<f64, U2>, epoch: Epoch) -> Self {
                BikeState { x: state, t: epoch }
            }
            pub fn zeros() -> Self {
                Self {
                    x: OVector::<f64, U2>::zeros(),
                    t: time::Epoch::from_gregorian_tai(2022, 5, 10, 0, 0, 0, 0),
                }
            }
        }

        let mut rng = rand::thread_rng();

        let s = OVector::<f64, U2>::new(-5.0, 1.0);
        let t = Epoch::now().unwrap();
        let mut bike_actual = BikeState::new(s, t);
        let dynamics = |x: &OVector<f64, U2>, _ext: &OVector<f64, Const<1>>, dt: Duration| {
            OVector::<f64, U2>::new(x[0] + x[1] * dt.in_seconds(), x[1])
        };
        let measure_model = |x: &OVector<f64, U2>| OVector::<f64, U2>::new(x[0], x[1]);
        let mut add_noisies = |mut y: OVector<f64, U2>| {
            y[0] += rng.gen_range(-3.0..3.0);
            y[1] += rng.gen_range(-0.1..0.1);
            y
        };
        let mut actual_normed_noise: Vec<f64> = Vec::new();
        let mut estimate_normed_error: Vec<f64> = Vec::new();

        let mut ukf = UKF::<U2, Const<4>, U2, Const<1>, BikeState>::build(
            Box::new(dynamics),
            Box::new(measure_model),
            Box::new(MinimalSkewSimplexSampling::build(0.6).unwrap()),
            BikeState::zeros(),
            OMatrix::<f64, U2, U2>::from_diagonal_element(10.0),
            OMatrix::<f64, U2, U2>::from_diagonal_element(1.0),
            OMatrix::<f64, U2, U2>::from_diagonal(&OVector::<f64, U2>::new(1.0, 0.001)),
        );
        // libc_println!(
        //     "bike actual = {:?}, ukf estimate = {:?}",
        //     &bike_actual,
        //     &ukf.current_estimate()
        // );

        let ukf_base_epoch = ukf.current_estimate().epoch();
        let nums_measure = 500_usize;
        for i in 0..nums_measure {
            let dt = Duration::from_f64(1.0, Unit::Second);
            let m_epoch = ukf_base_epoch + dt;
            let _ = bike_actual.propagate(&dynamics, dt, OVector::<f64, Const<1>>::zeros());
            let mut meas = measure_model(&bike_actual.state());
            meas = add_noisies(meas);
            ukf.feed_and_update(meas, m_epoch, OVector::<f64, Const<1>>::zeros());

            if i > nums_measure / 3 {
                actual_normed_noise.push((&meas - bike_actual.state()).norm());
                estimate_normed_error
                    .push((ukf.current_estimate().state() - bike_actual.state()).norm());
            }
        }
        let nums = actual_normed_noise.len();
        let noise_metric: f64 = actual_normed_noise
            .into_iter()
            .fold(0.0, |acc, x| acc + x / nums as f64);
        let error_metric: f64 = estimate_normed_error
            .into_iter()
            .fold(0.0, |acc, x| acc + x / nums as f64);

        assert!(error_metric < noise_metric);
        // libc_println!("error_metric = {:?}", error_metric);
        // libc_println!("noise_metric = {:?}", noise_metric);
    }

    #[test]
    fn test_dynamic_matrix() {
        extern crate libc_print;
        use crate::dfilters::dstate::test;
        use libc_print::libc_println;
        let m = test();
        // libc_println!("m = {:?}", m);
    }
}
