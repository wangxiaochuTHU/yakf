use super::sigma_points::SamplingMethod;
use super::state::State;
use crate::alloc::boxed::Box;
use crate::errors::YakfError;
use crate::itertools::izip;
use crate::linalg::allocator::Allocator;
use crate::linalg::{DefaultAllocator, DimName, OMatrix, OVector};
use crate::time::{Duration, Epoch};

/// `T`: state dimension, e.g., `Const<6>`
///
/// `T2`:Samples number (e.g, T2 = T + 2 for Minimal skew simplex sampling method)
///
/// `M`: measuremtent dimension
///
/// `U`: External vector dimension
///
/// `S`: A state that implements `State` trait
#[allow(dead_code)]
pub struct UKF<T, T2, M, U, S>
where
    T: DimName,
    T2: DimName,
    M: DimName,
    U: DimName,
    S: State<T, U>,
    DefaultAllocator: Allocator<f64, T>
        + Allocator<f64, T, T>
        + Allocator<f64, M, M>
        + Allocator<f64, T>
        + Allocator<f64, U>
        + Allocator<f64, U>
        + Allocator<f64, T2>
        + Allocator<f64, M>
        + Allocator<f64, na::Const<1_usize>, T>
        + Allocator<f64, na::Const<1_usize>, M>
        + Allocator<f64, T, M>
        + Allocator<f64, M, T>
        + Allocator<f64, T, T2>
        + Allocator<f64, M, T2>,
{
    pub dynamics: Box<dyn Fn(&OVector<f64, T>, &OVector<f64, U>, Duration) -> OVector<f64, T>>,
    pub measure_model: Box<dyn Fn(&OVector<f64, T>) -> OVector<f64, M>>,
    pub sampling: Box<dyn SamplingMethod<T, T2>>,

    pub prev_x: S,
    pub prev_p: OMatrix<f64, T, T>,
    pub process_q: OMatrix<f64, T, T>,
    pub process_r: OMatrix<f64, M, M>,
}
impl<T, T2, M, U, S> UKF<T, T2, M, U, S>
where
    T: DimName,
    T2: DimName,
    M: DimName,
    U: DimName,
    S: State<T, U>,
    DefaultAllocator: Allocator<f64, T>
        + Allocator<f64, T, T>
        + Allocator<f64, M, M>
        + Allocator<f64, T>
        + Allocator<f64, U>
        + Allocator<f64, U>
        + Allocator<f64, T2>
        + Allocator<f64, M>
        + Allocator<f64, na::Const<1_usize>, T>
        + Allocator<f64, na::Const<1_usize>, M>
        + Allocator<f64, T, M>
        + Allocator<f64, M, T>
        + Allocator<f64, T, T2>
        + Allocator<f64, M, T2>,
{
    #[allow(dead_code)]
    /// function that returns a UKF
    pub fn build(
        dynamics: Box<dyn Fn(&OVector<f64, T>, &OVector<f64, U>, Duration) -> OVector<f64, T>>,
        measure_model: Box<dyn Fn(&OVector<f64, T>) -> OVector<f64, M>>,
        sampling: Box<dyn SamplingMethod<T, T2>>,
        prev_x: S,
        prev_p: OMatrix<f64, T, T>,
        process_q: OMatrix<f64, T, T>,
        process_r: OMatrix<f64, M, M>,
    ) -> Self {
        Self {
            dynamics,
            measure_model,
            sampling,
            prev_x,
            prev_p,
            process_q,
            process_r,
        }
    }

    #[allow(dead_code)]
    /// resets state to a new one
    pub fn reset_state(&mut self, new_s: S) {
        self.prev_x = new_s;
    }

    #[allow(dead_code)]
    pub fn reset_p(&mut self, new_p: OMatrix<f64, T, T>) {
        self.prev_p = new_p;
    }

    #[allow(dead_code)]
    pub fn reset_r(&mut self, new_r: OMatrix<f64, M, M>) {
        self.process_r = new_r;
    }

    #[allow(dead_code)]
    pub fn reset_q(&mut self, new_q: OMatrix<f64, T, T>) {
        self.process_q = new_q;
    }

    #[allow(dead_code)]
    pub fn current_estimate(&self) -> &S {
        &self.prev_x
    }

    #[allow(dead_code)]
    pub fn samples_dimention(&self) -> usize {
        T2::dim()
    }

    #[allow(dead_code)]
    fn state_samples(&self) -> Result<OMatrix<f64, T, T2>, YakfError> {
        match (&self.prev_p + &self.process_q).cholesky() {
            Some(cholesky) => {
                let cho = cholesky.unpack();
                let mut samples: OMatrix<f64, T, T2> = OMatrix::<f64, T, T2>::zeros();
                let bases = self.sampling.bases();
                for (i, mut col) in samples.column_iter_mut().enumerate() {
                    let u_i = &bases.column(i);
                    let chi = &self.prev_x.state() + &cho * u_i;
                    col.copy_from(&chi);
                }

                // for (i, u) in self.sampling.bases().iter().enumerate() {
                //     let chi = self.prev_x.state() + &cho * u;
                //     samples.push(chi);
                // }
                Ok(samples)
            }
            None => Err(YakfError::CholeskyErr),
        }
    }

    #[allow(dead_code)]
    pub fn feed_and_update(
        &mut self,
        measure: OVector<f64, M>,
        m_epoch: Epoch,
        external: OVector<f64, U>,
    ) {
        match self.state_samples() {
            Ok(samples) => {
                // delta time, since previous estimate
                let dt = m_epoch - self.prev_x.epoch();
                // libc_println!("samples = {:?}", samples);

                // propagates sampling states.
                let mut samples_propagated = OMatrix::<f64, T, T2>::zeros();
                for (i, mut sp) in samples_propagated.column_iter_mut().enumerate() {
                    let s = &samples.column(i);
                    sp.copy_from(&self.propagate(s.into_owned(), dt, &external));
                }
                // libc_println!("samples_propagated = {:?}", samples_propagated);
                // panic!();

                // predicts theoretic measurements according to measuring model.
                let mut measures_predicted = OMatrix::<f64, M, T2>::zeros();
                for (i, mut mp) in measures_predicted.column_iter_mut().enumerate() {
                    let sp = &samples_propagated.column(i);
                    mp.copy_from(&self.measure(&sp.into_owned()));
                }

                // calculates the weighted average of propagated samples.
                let μ_x = samples_propagated
                    .column_iter()
                    .zip(self.sampling.weights_m().iter())
                    .fold(OVector::<f64, T>::zeros(), |acc, (ref x, w)| acc + *w * x);

                // calculates the weighted average of predicted measurements.
                let μ_y = measures_predicted
                    .column_iter()
                    .zip(self.sampling.weights_m().iter())
                    .fold(OVector::<f64, M>::zeros(), |acc, (ref x, w)| acc + *w * x);

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
                    .fold(OMatrix::<f64, M, M>::zeros(), |acc, (ref y, w)| {
                        acc + *w * (y - &μ_y) * (y - &μ_y).transpose()
                    });

                // predicts the correlative covariance.
                let mut p_xy_predicted = OMatrix::<f64, T, M>::zeros();
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
        state: OVector<f64, T>,
        dt: Duration,
        external: &OVector<f64, U>,
    ) -> OVector<f64, T> {
        (self.dynamics)(&state, external, dt)
    }

    #[allow(dead_code)]
    fn gain_factor(
        &self,
        p_xy: OMatrix<f64, T, M>,
        p_yy: &OMatrix<f64, M, M>,
    ) -> Result<OMatrix<f64, T, M>, YakfError> {
        match (p_yy + &self.process_r).try_inverse() {
            Some(inv) => Ok(p_xy * inv),
            None => Err(YakfError::InverseErr),
        }
    }
    fn measure(&self, state: &OVector<f64, T>) -> OVector<f64, M> {
        (self.measure_model)(state)
    }
}
