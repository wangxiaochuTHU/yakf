use crate::time::{Duration, Epoch};

use crate::alloc::{boxed::Box, vec::Vec};
use crate::errors::YakfError;
use crate::lie::base::{LieAlgebraSE3, LieGroupSE3, LieVectorSE3};
use crate::linalg::allocator::Allocator;
use crate::linalg::{Const, DefaultAllocator, DimName, OMatrix, OVector, U3, U4, U6};
pub trait ESStates {
    /// get the state
    fn state(&self) -> &LieGroupSE3 {
        unimplemented!()
    }

    /// set the state
    fn set_state(&mut self, state: LieGroupSE3) {
        unimplemented!()
    }

    /// get the epoch
    fn epoch(&self) -> Epoch {
        unimplemented!()
    }

    /// set the epoch
    fn set_epoch(&mut self, _epoch: Epoch) {
        unimplemented!()
    }

    // /// propagate / predict
    // fn propagate(
    //     &mut self,
    //     dynamics: &dyn Fn(&LieGroupSE3, &LieVectorSE3, Duration) -> LieGroupSE3,
    //     dt: Duration,
    //     external: LieVectorSE3,
    // ) {
    //     self.set_state(dynamics(&self.state(), &external, dt));
    //     self.set_epoch(self.epoch() + dt);
    // }
}

pub struct ESEKF<S>
where
    S: ESStates,
{
    pub stamp_state: S,
    pmatrix: OMatrix<f64, U6, U6>,
    qmatrix: OMatrix<f64, U6, U6>,
    nmatrix: OMatrix<f64, Const<12>, Const<12>>,
    f: Box<dyn Fn(&LieGroupSE3, &LieVectorSE3, Duration) -> OMatrix<f64, U6, U6>>,
    g: Box<dyn Fn(&LieGroupSE3, &LieVectorSE3, Duration) -> OMatrix<f64, U6, U6>>,
    h: Box<dyn Fn(&LieGroupSE3) -> OMatrix<f64, Const<12>, U6>>,
    ob: Box<dyn Fn(&LieGroupSE3) -> OVector<f64, Const<12>>>,
}
impl<S> ESEKF<S>
where
    S: ESStates,
{
    #[allow(dead_code)]
    /// function that returns a UKF
    pub fn build(
        f: Box<dyn Fn(&LieGroupSE3, &LieVectorSE3, Duration) -> OMatrix<f64, U6, U6>>,
        g: Box<dyn Fn(&LieGroupSE3, &LieVectorSE3, Duration) -> OMatrix<f64, U6, U6>>,
        h: Box<dyn Fn(&LieGroupSE3) -> OMatrix<f64, Const<12>, U6>>,
        ob: Box<dyn Fn(&LieGroupSE3) -> OVector<f64, Const<12>>>,
        stamp_state: S,
        pmatrix: OMatrix<f64, U6, U6>,
        qmatrix: OMatrix<f64, U6, U6>,
        nmatrix: OMatrix<f64, Const<12>, Const<12>>,
    ) -> Self {
        Self {
            stamp_state,
            pmatrix,
            qmatrix,
            nmatrix,
            f,
            g,
            h,
            ob,
        }
    }

    pub fn transition_f(
        &self,
        x_estimate: &LieGroupSE3,
        u: &LieVectorSE3,
        dt: Duration,
    ) -> OMatrix<f64, U6, U6> {
        (self.f)(x_estimate, u, dt)
    }

    pub fn transition_g(
        &self,
        x_estimate: &LieGroupSE3,
        u: &LieVectorSE3,
        dt: Duration,
    ) -> OMatrix<f64, U6, U6> {
        (self.g)(x_estimate, u, dt)
    }

    pub fn transition_h(&self, x_predict: &LieGroupSE3) -> OMatrix<f64, Const<12>, U6> {
        (self.h)(x_predict)
    }

    pub fn propagate(&self, u: &LieVectorSE3, dt: Duration) -> LieGroupSE3 {
        let u_col = u.to_vec6();
        let f = self.transition_f(self.stamp_state.state(), u, dt);

        let inc_col = f * u_col;
        let delta_alg = LieVectorSE3::from_vec6(&inc_col).to_algebra() * dt.in_seconds();
        let delta_group = LieGroupSE3::from_algebra(&delta_alg);
        let mut m = self.stamp_state.state().clone();
        m.increment_by_left_delta(delta_group);
        m
    }
    pub fn measure(&self, x: &LieGroupSE3) -> OVector<f64, Const<12>> {
        (self.ob)(x)
    }

    #[allow(dead_code)]
    pub fn feed_and_update(
        &mut self,
        measure: OVector<f64, Const<12>>,
        m_epoch: Epoch,
        u: LieVectorSE3,
    ) -> Result<(), YakfError> {
        let dt = m_epoch - self.stamp_state.epoch();

        let mut x_predict = self.propagate(&u, dt);

        let f = self.transition_f(self.stamp_state.state(), &u, dt);
        let g = self.transition_g(self.stamp_state.state(), &u, dt);

        let p_predict = f * &self.pmatrix * &f.transpose() + g * &self.qmatrix * &g.transpose();

        let ob_predict = self.measure(&x_predict);

        let z = measure - ob_predict;

        let h = self.transition_h(&x_predict);

        let zmatrix = h * p_predict * h.transpose() + self.nmatrix;

        match zmatrix.try_inverse() {
            Some(zm_inv) => {
                let kmatrix = p_predict * h.transpose() * zm_inv;
                let dx = kmatrix * z;
                let dx_group = LieVectorSE3::from_vec6(&dx).to_group();
                x_predict.increment_by_left_delta(dx_group);
                self.stamp_state.set_state(x_predict);
                self.stamp_state.set_epoch(m_epoch);
                self.pmatrix = &self.pmatrix - &kmatrix * &zmatrix * &kmatrix.transpose();

                Ok(())
            }
            None => Err(YakfError::InverseErr),
        }
        // let k = p_predict*h.transpose()*
        // let g_x = self.transition_f(&x_predict, dt);
        // let p_predict = &g_x * &self.prev_p * g_x.transpose() + &self.process_q;

        // let h_x = self.transition_h(&x_predict);
        // match (&h_x * &p_predict * &h_x.transpose() + &self.process_r).try_inverse() {
        //     Some(inv) => {
        //         let k = &p_predict * h_x.transpose() * inv;
        //         let new_estimate = x_predict + &k * (measure - z_predict);
        //         self.prev_x.set_state(new_estimate);
        //         self.prev_x.set_epoch(m_epoch);
        //         let sub = dmatrix_identity(self.n, self.n) - &k * h_x;
        //         self.prev_p = &sub * &p_predict * &sub.transpose() + &self.process_r;
        //         Ok(())
        //     }
        //     None => Err(YakfError::InverseErr),
        // }
    }
}
