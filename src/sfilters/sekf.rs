use crate::time::{Duration, Epoch};

use crate::alloc::{boxed::Box, vec::Vec};
use crate::lie::base::{LieAlgebraSE3, LieGroupSE3, LieVectorSE3};
use crate::linalg::allocator::Allocator;
use crate::linalg::{DefaultAllocator, DimName, OMatrix, OVector, U3, U4, U6};

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

    /// propagate / predict
    fn propagate(
        &mut self,
        dynamics: &dyn Fn(&LieGroupSE3, &LieVectorSE3, Duration) -> LieGroupSE3,
        dt: Duration,
        external: LieVectorSE3,
    ) {
        self.set_state(dynamics(&self.state(), &external, dt));
        self.set_epoch(self.epoch() + dt);
    }
}

pub struct ESEKF<S>
where
    S: ESStates,
{
    stamp_state: S,
    pmatrix: OMatrix<f64, U6, U6>,
    qmatrix: OMatrix<f64, U6, U6>,
    nmatrix: OMatrix<f64, U3, U3>,
    f: Box<dyn Fn(&LieGroupSE3, &LieVectorSE3, Duration) -> OMatrix<f64, U6, U6>>,
    g: Box<dyn Fn(&LieGroupSE3, &LieVectorSE3, Duration) -> OMatrix<f64, U6, U6>>,
}
