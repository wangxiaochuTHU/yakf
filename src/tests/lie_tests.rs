// #[cfg(test)]
// mod tests {

//     use crate::lie::base::{LieAlgebraSE3, LieGroupSE3, LieVectorSE3};

//     use crate::linalg::{Const, OMatrix, OVector, U2, U3};
//     /// import Re-exports of hifitime (for time) and nalgebra (for matrix)
//     use crate::{
//         linalg,
//         time::{Duration, Epoch, Unit},
//     };
//     #[test]
//     fn test_vec6_group() {
//         let w = OVector::<f64, U3>::new(1.2, 2.5, 3.7);
//         let w = w / w.norm();
//         let vec6 = LieVectorSE3 {
//             w: w,
//             v: OVector::<f64, U3>::new(5.6, 6.7, 7.8),
//         };

//         let group = LieGroupSE3::from(&vec6);

//         let vec6_back = group.to_vec6();

//         assert!((&vec6_back.w - &vec6.w).norm() < 1e-10);
//         assert!((&vec6_back.v - &vec6.v).norm() < 1e-10);
//     }

//     #[test]
//     fn test_vec6_algebra_group() {
//         let ws = [
//             OVector::<f64, U3>::new(1.2, 2.5, 3.7),
//             OVector::<f64, U3>::new(-1.2, -2.5, 3.7),
//             OVector::<f64, U3>::new(1e-8, 1e-8, 1e-8),
//         ];

//         for w in ws.into_iter() {
//             let w = OVector::<f64, U3>::new(1.2, 2.5, 3.7);
//             let w = w / w.norm();
//             let vec6 = LieVectorSE3 {
//                 w: w,
//                 v: OVector::<f64, U3>::new(5.6, 6.7, 7.8),
//             };
//             let hat4 = vec6.to_algebra();
//             let group = LieGroupSE3::from_algebra(&hat4);
//             let hat4_back = group.to_algebra();
//             let vec6_back = LieVectorSE3::from_algebra(&hat4_back);
//             assert!((&vec6_back.w - &vec6.w).norm() < 1e-10);
//             assert!((&vec6_back.v - &vec6.v).norm() < 1e-10);
//             assert!((&hat4_back - hat4).norm() < 1e-10);
//         }
//     }
//     #[test]
//     fn test_adjoint_matrix() {
//         let w = OVector::<f64, U3>::new(1.2, 2.5, 3.7);
//         let w = w / w.norm();
//         let vec6 = LieVectorSE3 {
//             w: w,
//             v: OVector::<f64, U3>::new(5.6, 6.7, 7.8),
//         };
//         let group = LieGroupSE3::from(&vec6);

//         let tau = OVector::<f64, U3>::new(5.2, 5.5, 2.7);
//         let tau = w / w.norm();
//         let tau_vec = LieVectorSE3 {
//             w: w,
//             v: OVector::<f64, U3>::new(1.6, 2.7, 3.8),
//         };
//         let tau_hat = tau_vec.to_algebra();

//         // 1. use adjoint_action
//         let tau_hat_i = group.adjoint_action(&tau_hat);
//         let tau_vec_i = LieVectorSE3::from_algebra(&tau_hat_i);

//         // 2. use adjoint matrix
//         let adj = group.adjoint_matrix();
//         let tau_column = tau_vec.to_vec6();
//         let tau_column_j = adj * tau_column;
//         let tau_vec_j = LieVectorSE3::from_vec6(&tau_column_j);
//         let tau_hat_j = tau_vec_j.to_algebra();

//         assert!((&tau_hat_i - &tau_hat_j).norm() < 1e-10);
//         assert!((&tau_vec_i.w - &tau_vec_j.w).norm() < 1e-10);
//         assert!((&tau_vec_i.v - &tau_vec_j.v).norm() < 1e-10);
//     }

//     #[test]
//     fn test_action_point() {
//         let w = OVector::<f64, U3>::new(0.0, 0.0, 0.0);
//         let w = if w.norm() != 0.0 { w / w.norm() } else { w };

//         let vec6 = LieVectorSE3 {
//             w: w,
//             v: OVector::<f64, U3>::new(1.0, 2.0, 3.0),
//         };

//         let group = LieGroupSE3::from(&vec6);

//         let p = OVector::<f64, U3>::new(3.0, 4.0, 5.0);

//         let p2 = group.action_on_point(&p);
//         let p2_expect = OVector::<f64, U3>::new(4.0, 6.0, 8.0);

//         assert!((&p2 - p2_expect).norm() < 1e-10);
//     }
//     #[test]
//     fn test_action_around_circle() {
//         let radius = 100.0;
//         let half_pi = core::f64::consts::PI / 2.0;
//         let mut pose_group = LieGroupSE3::from_r_t(
//             OMatrix::<f64, U3, U3>::identity(),
//             OVector::<f64, U3>::new(radius, 0.0, 0.0),
//         );
//         let expected_head_global = [
//             OVector::<f64, U3>::new(100.0, 1.0, 0.0),
//             OVector::<f64, U3>::new(-1.0, 100.0, 0.0),
//             OVector::<f64, U3>::new(-100.0, -1.0, 0.0),
//             OVector::<f64, U3>::new(1.0, -100.0, 0.0),
//         ];

//         let expected_head_vec_global = [
//             OVector::<f64, U3>::new(0.0, 1.0, 0.0),
//             OVector::<f64, U3>::new(-1.0, 0.0, 0.0),
//             OVector::<f64, U3>::new(0.0, -1.0, 0.0),
//             OVector::<f64, U3>::new(1.0, 0.0, 0.0),
//         ];

//         for i in 0..4 {
//             let angle = half_pi * i as f64;
//             let x = radius * angle.cos();
//             let y = radius * angle.sin();
//             let z = 0.0;
//             let t = OVector::<f64, U3>::new(x, y, z);
//             let ebi = OVector::<f64, U3>::new(angle.cos(), angle.sin(), 0.0);
//             let ebj = OVector::<f64, U3>::new(-angle.sin(), angle.cos(), 0.0);
//             let ebk = OVector::<f64, U3>::new(0.0, 0.0, 1.0);
//             let r = OMatrix::<f64, U3, U3>::from_columns(&[ebi, ebj, ebk]);

//             // the pose obtained based on geometry
//             let geometry_pose_group = LieGroupSE3::from_r_t(r, t);
//             let head_local = OVector::<f64, U3>::new(0.0, 1.0, 0.0);
//             let head_global = geometry_pose_group.action_on_point(&head_local);

//             let head_vec_local = OVector::<f64, U3>::new(0.0, 1.0, 0.0);
//             let head_alg_local = LieVectorSE3 {
//                 w: OVector::<f64, U3>::zeros(),
//                 v: head_vec_local,
//             }
//             .to_algebra();
//             let head_alg_global = geometry_pose_group.adjoint_action(&head_alg_local);
//             let head_vec_global = LieVectorSE3::from_algebra(&head_alg_global).v;

//             let head_global_error = head_global - expected_head_global[i];
//             let head_vec_global_error = head_vec_global - expected_head_vec_global[i];

//             assert!(head_global_error.norm() < 1e-10);
//             assert!(head_vec_global_error.norm() < 1e-10);
//         }
//     }
// }
