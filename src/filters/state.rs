use crate::time::{Duration, Epoch};

use crate::linalg::allocator::Allocator;
use crate::linalg::{DefaultAllocator, DimName, OVector};

/// `State` trait.
///
/// `T`: state vector dimension. e.g., `Const<3>` stands for 3-D
///
/// `U`: external vector dimension (You should assign any DimName to it, though it is not in-use currently)
pub trait State<T, U>
where
    T: DimName,
    U: DimName,
    DefaultAllocator: Allocator<f64, U> + Allocator<f64, T>,
{
    /// get the state vec
    fn state(&self) -> OVector<f64, T> {
        unimplemented!()
    }

    /// set the state vec
    fn set_state(&mut self, _state: OVector<f64, T>) {
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

    /// propagate
    fn propagate(
        &mut self,
        dynamics: &dyn Fn(&OVector<f64, T>, &OVector<f64, U>, Duration) -> OVector<f64, T>,
        dt: Duration,
        external: OVector<f64, U>,
    ) {
        self.set_state(dynamics(&self.state(), &external, dt));
        self.set_epoch(self.epoch() + dt);
    }
}
