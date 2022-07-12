use crate::time::{Duration, Epoch};

use crate::linalg::{DMatrix, DVector};

/// `DState` trait. dynamically-sized version of `State`

pub trait DState {
    /// get the state vec
    fn state(&self) -> &DVector<f64> {
        unimplemented!()
    }

    /// set the state vec
    fn set_state(&mut self, _state: DVector<f64>) {
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
        dynamics: &dyn Fn(&DVector<f64>, &DVector<f64>, Duration) -> DVector<f64>,
        dt: Duration,
        external: DVector<f64>,
    ) {
        self.set_state(dynamics(&self.state(), &external, dt));
        self.set_epoch(self.epoch() + dt);
    }
}

#[allow(dead_code)]
pub fn test() -> DMatrix<f64> {
    use alloc::vec::Vec;
    pub struct BikeState {
        pub x: DVector<f64>,
        pub t: Epoch,
    }

    /// for example, you can define your own methods.
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
    impl BikeState {
        pub fn empty(d: usize) -> Self {
            Self {
                x: DVector::<f64>::zeros(d),
                t: Epoch::from_gpst_nanoseconds(0),
            }
        }
    }
    let mut v1 = Vec::<f64>::new();
    v1.push(1.0);
    v1.push(2.0);
    v1.push(3.0);

    let dv1 = DVector::from_row_slice(v1.as_slice());
    let dv = DVector::from_row_slice(&[11.0, 21.0, 31.0]);
    let m = DMatrix::<f64>::from_columns(&[dv1, dv]);
    let _dbike_1d = BikeState::empty(2);
    let _dbike_3d = BikeState::empty(6);

    m
}
