use super::dfixed::dmatrix_identity;

use super::dstate::DState;
use crate::alloc::boxed::Box;
use crate::errors::YakfError;

use crate::linalg::{DMatrix, DVector};
use crate::time::{Duration, Epoch};

/*
    If you want to clone the KF, the following trait may help.
    This is studied from the reference
[https://stackoverflow.com/questions/65203307/how-do-i-create-a-trait-object-that-implements-fn-and-can-be-cloned-to-distinct]

    after implementing CloneableFn, you should need to modify the signature of DEKF accordingly. e.g.
    change this line  -->  pub dynamics: Box<dyn Fn(&DVector<f64>, &DVector<f64>, Duration) -> DVector<f64>>,
                    to-->  pub dynamics: Box<dyn CloneableFn>,
*/

// trait CloneableFn: Fn(&DVector<f64>, &DVector<f64>, Duration) -> DVector<f64> {
//     fn clone_box<'a>(&self) -> Box<dyn 'a + CloneableFn>
//     where
//         Self: 'a;
// }

// impl<F: Fn(&DVector<f64>, &DVector<f64>, Duration) -> DVector<f64> + Clone> CloneableFn for F {
//     fn clone_box<'a>(&self) -> Box<dyn 'a + CloneableFn>
//     where
//         Self: 'a,
//     {
//         Box::new(self.clone())
//     }
// }

// impl<'a> Clone for Box<dyn 'a + CloneableFn> {
//     fn clone(&self) -> Self {
//         (**self).clone_box()
//     }
// }

// #[derive(Clone)]

pub struct DEKF<S>
where
    S: DState,
{
    // state dimension
    pub n: usize,
    // measure dimension
    pub m: usize,
    pub dynamics: Box<dyn Fn(&DVector<f64>, &DVector<f64>, Duration) -> DVector<f64>>,
    pub g: Box<dyn Fn(&DVector<f64>, Duration) -> DMatrix<f64>>,
    pub measure_model: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
    pub h: Box<dyn Fn(&DVector<f64>) -> DMatrix<f64>>,
    pub prev_x: S,
    pub prev_p: DMatrix<f64>,
    pub process_q: DMatrix<f64>,
    pub process_r: DMatrix<f64>,
}

impl<S> DEKF<S>
where
    S: DState,
{
    #[allow(dead_code)]
    /// function that returns a DEKF
    /// n: state dimension, m: measurement dimension
    /// dynamics: old state -> new state dynamics
    /// g: transition matrix with 1st order, derived from dynamics
    /// measure_model: state -> observation model
    /// h: measure_model transition matrix with 1st order, derived from observation
    ///
    pub fn build(
        n: usize,
        m: usize,
        dynamics: Box<dyn Fn(&DVector<f64>, &DVector<f64>, Duration) -> DVector<f64>>,
        g: Box<dyn Fn(&DVector<f64>, Duration) -> DMatrix<f64>>,
        measure_model: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
        h: Box<dyn Fn(&DVector<f64>) -> DMatrix<f64>>,
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
                m,
                dynamics,
                g,
                measure_model,
                h,
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

    pub fn transition_f(&self, x_predict: &DVector<f64>, dt: Duration) -> DMatrix<f64> {
        (self.g)(x_predict, dt)
    }

    pub fn transition_h(&self, x_predict: &DVector<f64>) -> DMatrix<f64> {
        (self.h)(x_predict)
    }

    #[allow(dead_code)]
    pub fn feed_and_update(
        &mut self,
        measure: DVector<f64>,
        m_epoch: Epoch,
        external: DVector<f64>,
    ) -> Result<(), YakfError> {
        let dt = m_epoch - self.prev_x.epoch();
        let x_predict = self.propagate(self.prev_x.state(), dt, &external);
        let z_predict = self.measure(&x_predict);
        let g_x = self.transition_f(&x_predict, dt);
        let p_predict = &g_x * &self.prev_p * g_x.transpose() + &self.process_q;

        let h_x = self.transition_h(&x_predict);
        match (&h_x * &p_predict * &h_x.transpose() + &self.process_r).try_inverse() {
            Some(inv) => {
                let k = &p_predict * h_x.transpose() * inv;
                let new_estimate = x_predict + &k * (measure - z_predict);
                self.prev_x.set_state(new_estimate);
                self.prev_x.set_epoch(m_epoch);
                let sub = dmatrix_identity(self.n, self.n) - &k * h_x;
                self.prev_p = &sub * &p_predict * &sub.transpose() + &self.process_r;
                Ok(())
            }
            None => Err(YakfError::InverseErr),
        }
    }

    #[allow(dead_code)]
    fn propagate(
        &self,
        state: &DVector<f64>,
        dt: Duration,
        external: &DVector<f64>,
    ) -> DVector<f64> {
        (self.dynamics)(&state, external, dt)
    }

    fn measure(&self, state: &DVector<f64>) -> DVector<f64> {
        (self.measure_model)(state)
    }
}
