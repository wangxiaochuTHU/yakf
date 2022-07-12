use super::dfixed::{dmatrix_zeros, dvector_zeros};
use super::dsigma_points::DSamplingMethod;
use super::dstate::DState;
use crate::alloc::boxed::Box;
use crate::errors::YakfError;
use crate::itertools::izip;
use crate::linalg::{DMatrix, DVector};
use crate::time::{Duration, Epoch};

/// Dynamically-sized version of `UKF`
/// `S`: A state that implements `DState` trait
///
#[allow(dead_code)]
pub struct DUKF<S>
where
    S: DState,
{
    // state dimension
    pub n: usize,
    // samples numbers
    pub n2: usize,
    // measure dimension
    pub m: usize,
    pub dynamics: Box<dyn Fn(&DVector<f64>, &DVector<f64>, Duration) -> DVector<f64>>,
    pub measure_model: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
    pub sampling: Box<dyn DSamplingMethod>,

    pub prev_x: S,
    pub prev_p: DMatrix<f64>,
    pub process_q: DMatrix<f64>,
    pub process_r: DMatrix<f64>,
}
impl<S> DUKF<S>
where
    S: DState,
{
    #[allow(dead_code)]
    /// function that returns a UKF
    pub fn build(
        n: usize,
        n2: usize,
        m: usize,
        dynamics: Box<dyn Fn(&DVector<f64>, &DVector<f64>, Duration) -> DVector<f64>>,
        measure_model: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
        sampling: Box<dyn DSamplingMethod>,
        prev_x: S,
        prev_p: DMatrix<f64>,
        process_q: DMatrix<f64>,
        process_r: DMatrix<f64>,
    ) -> Result<Self, YakfError> {
        
        if prev_x.state().len() != n {
            Err(YakfError::DimensionMismatchErr)
        } else {
            Ok(Self {
                n,
                n2,
                m,
                dynamics,
                measure_model,
                sampling,
                prev_x,
                prev_p,
                process_q,
                process_r,
            })
        }
    }

    #[allow(dead_code)]
    /// resets state to a new one
    pub fn reset_state(&mut self, new_s: S) {
        self.prev_x = new_s;
    }

    #[allow(dead_code)]
    pub fn reset_p(&mut self, new_p: DMatrix<f64>) {
        self.prev_p = new_p;
    }

    #[allow(dead_code)]
    pub fn reset_r(&mut self, new_r: DMatrix<f64>) {
        self.process_r = new_r;
    }

    #[allow(dead_code)]
    pub fn reset_q(&mut self, new_q: DMatrix<f64>) {
        self.process_q = new_q;
    }

    #[allow(dead_code)]
    pub fn current_estimate(&self) -> &S {
        &self.prev_x
    }

    #[allow(dead_code)]
    pub fn samples_dimension(&self) -> usize {
        self.n2
    }

    #[allow(dead_code)]
    fn state_samples(&self) -> Result<DMatrix<f64>, YakfError> {
        self.sampling
            .sampling_states(&self.prev_p, &self.prev_x.state())
    }

    #[allow(dead_code)]
    pub fn feed_and_update(
        &mut self,
        measure: DVector<f64>,
        m_epoch: Epoch,
        external: DVector<f64>,
    ) {
        match self.state_samples() {
            Ok(samples) => {
                // delta time, since previous estimate
                let dt = m_epoch - self.prev_x.epoch();
                // libc_println!("samples = {:?}", samples);
                // libc_println!("samples.len = {:?}", samples);
                // panic!();
                // propagates sampling states.
                let mut samples_propagated = dmatrix_zeros(self.n, self.n2);
                for (i, mut sp) in samples_propagated.column_iter_mut().enumerate() {
                    let s = &samples.column(i);
                    sp.copy_from(&self.propagate(s.into_owned(), dt, &external));
                }
                // panic!();
                // libc_println!("samples_propagated = {:?}", samples_propagated);
                // panic!();

                // predicts theoretic measurements according to measuring model.
                let mut measures_predicted = dmatrix_zeros(self.m, self.n2);
                for (i, mut mp) in measures_predicted.column_iter_mut().enumerate() {
                    let sp = &samples_propagated.column(i);
                    mp.copy_from(&self.measure(&sp.into_owned()));
                }

                // calculates the weighted average of propagated samples.
                let μ_x = samples_propagated
                    .column_iter()
                    .zip(self.sampling.weights_m().iter())
                    .fold(dvector_zeros(self.n), |acc, (ref x, w)| acc + *w * x);

                // calculates the weighted average of predicted measurements.
                let μ_y = measures_predicted
                    .column_iter()
                    .zip(self.sampling.weights_m().iter())
                    .fold(dvector_zeros(self.m), |acc, (ref x, w)| acc + *w * x);

                // predicts the state propagation error covariance.
                let p_xx_predicted = samples_propagated
                    .column_iter()
                    .zip(self.sampling.weights_c().iter())
                    .fold(self.process_q.clone(), |acc, (ref x, w)| {
                        acc + *w * (x - &μ_x) * (x - &μ_x).transpose()
                    });

                // predicts the measurement prediction error covariance.
                let p_yy_predicted = measures_predicted
                    .column_iter()
                    .zip(self.sampling.weights_c().iter())
                    .fold(dmatrix_zeros(self.m, self.m), |acc, (ref y, w)| {
                        acc + *w * (y - &μ_y) * (y - &μ_y).transpose()
                    });

                // predicts the correlative covariance.
                let mut p_xy_predicted = dmatrix_zeros(self.n, self.m);
                for (i, (ref x, ref y)) in izip!(
                    samples_propagated.column_iter(),
                    measures_predicted.column_iter(),
                )
                .enumerate()
                {
                    p_xy_predicted +=
                        self.sampling.weights_c()[i] * (x - &μ_x) * (y - &μ_y).transpose();
                }

                // calculates the gain factor, used for updating
                match self.gain_factor(p_xy_predicted, &p_yy_predicted) {
                    Ok(ref kai) => {
                        let new_state = μ_x + kai * (measure - μ_y);
                        self.prev_x.set_state(new_state);
                        self.prev_x.set_epoch(m_epoch);
                        self.prev_p = p_xx_predicted
                            - kai * (p_yy_predicted + &self.process_r) * kai.transpose();
                    }
                    Err(e) => error!("Error occurs in update: {:?}", e),
                }
            }
            Err(e) => error!("Error occurs in update: {:?}", e),
        }
    }

    #[allow(dead_code)]
    fn propagate(
        &self,
        state: DVector<f64>,
        dt: Duration,
        external: &DVector<f64>,
    ) -> DVector<f64> {
        (self.dynamics)(&state, external, dt)
    }

    #[allow(dead_code)]
    fn gain_factor(
        &self,
        p_xy: DMatrix<f64>,
        p_yy: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, YakfError> {
        match (p_yy + &self.process_r).try_inverse() {
            Some(inv) => Ok(p_xy * inv),
            None => Err(YakfError::InverseErr),
        }
    }
    fn measure(&self, state: &DVector<f64>) -> DVector<f64> {
        (self.measure_model)(state)
    }
}
