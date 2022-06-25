/// import yakf crate
extern crate yakf;
/// import State trait, UKF filter struct, and MSSS sampling method struct
use yakf::lie::base::{LieGroupSE3, LieVectorSE3};

use yakf::linalg::{Const, OMatrix, OVector, U2, U3};
/// import Re-exports of hifitime (for time) and nalgebra (for matrix)
use yakf::{
    linalg,
    time::{Duration, Epoch, Unit},
};
fn main() {
    let w = OVector::<f64, U3>::new(1.2, 2.5, 3.7);
    let w = w / w.norm();
    let vec1 = LieVectorSE3 {
        w: w,
        v: OVector::<f64, U3>::new(5.6, 6.7, 7.8),
    };
    println!("vec1 = {:?}", vec1);
    let group1 = LieGroupSE3::from(vec1);
    println!("group = {:#?}", group1);
    let vec1_invert = group1.to_algebra();
    println!("vec1_invert = {:?}", vec1_invert.unwrap());
}
