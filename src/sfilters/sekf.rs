use crate::time::{Duration, Epoch};

use crate::alloc::{boxed::Box, vec::Vec};
use crate::lie::base::{LieAlgebraSE3, LieGroupSE3, LieVectorSE3};
use crate::linalg::allocator::Allocator;
use crate::linalg::{DefaultAllocator, DimName, OVector};

pub trait ESStates {
    /// get the i-th state
    fn state(&self, i: usize) -> &LieGroupSE3 {
        unimplemented!()
    }

    /// get all states
    fn states_all(&self) -> &Vec<LieGroupSE3> {
        unimplemented!()
    }

    /// set the i-th state
    fn set_state(&mut self, i: usize, state: LieGroupSE3) {
        unimplemented!()
    }

    /// set all states
    fn set_states_all(&mut self, states: Vec<LieGroupSE3>) {
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
        dynamics: &dyn Fn(&Vec<LieGroupSE3>, &LieVectorSE3, Duration) -> Vec<LieGroupSE3>,
        dt: Duration,
        external: LieVectorSE3,
    ) {
        self.set_states_all(dynamics(&self.states_all(), &external, dt));
        self.set_epoch(self.epoch() + dt);
    }
}

pub struct ESEKF<S>
where
    S: ESStates,
{
    stamp_state: S,
}
