// use crate::time::{Duration, Epoch};

// use super::state::State;
// use crate::errors::YakfError;
// use crate::generic_array::{ArrayLength, GenericArray};
// use crate::linalg::allocator::Allocator;
// use crate::linalg::{
//     DefaultAllocator, DimName, OMatrix, OVector, RealField, VectorN, VectorSlice, U3,
// };

// pub trait Estimate<S>
// where
//     S: State,
//     DefaultAllocator:
//         Allocator<f64, <S as State>::StateLength> + Allocator<f64, <S as State>::ExternalLength>,
// {
//     /// length of the state vec
//     type DimX: DimName;

//     /// length of the external vec
//     type DimU: DimName;

//     /// length of the observation vec
//     type DimZ: DimName;

//     fn state(&self) -> S {
//         unimplemented!()
//     }

//     fn epoch(&self) -> Epoch {
//         self.state().epoch()
//     }

//     fn set_epoch(&mut self, epoch: Epoch) {
//         self.state().set_epoch(epoch);
//     }

//     // fn propagate(
//     //     &self,
//     //     dt: Duration,
//     //     dynamics: Box<dyn Fn(&OVector<T, DimX>, &OVector<T, DimU>) -> VectorN<T, DimZ>>,
//     //     external: OVector<T, Self::ExternalLength>,
//     // ) -> Result<(), YakfError> {
//     //     unimplemented!()
//     // }
// }

// // fn propagate(&mut self,dt: Duration,
// //     dynamics:
// //     external: OVector<T, Self::ExternalLength>) -> Result<(), YakfError> {
// //     unimplemented!()
// // }
