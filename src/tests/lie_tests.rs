#[cfg(test)]
mod tests {
    use crate::lie::base::{LieGroupSE3, LieVectorSE3};

    use crate::linalg::{Const, OMatrix, OVector, U2, U3};
    /// import Re-exports of hifitime (for time) and nalgebra (for matrix)
    use crate::{
        linalg,
        time::{Duration, Epoch, Unit},
    };
    #[test]
    fn test_vec6_to_group() {
        let w = OVector::<f64, U3>::new(1.2, 2.5, 3.7);
        let w = w / w.norm();
        let vec6 = LieVectorSE3 {
            w: w,
            v: OVector::<f64, U3>::new(5.6, 6.7, 7.8),
        };

        let group = LieGroupSE3::from(&vec6);

        let vec6_back = group.to_algebra().unwrap();

        assert!((&vec6_back.w - &vec6.w).norm() < 1e-10);
        assert!((&vec6_back.v - &vec6.v).norm() < 1e-10);
    }

    #[test]
    fn test_vec6_hat_group() {
        let w = OVector::<f64, U3>::new(1.2, 2.5, 3.7);
        let w = w / w.norm();
        let vec6 = LieVectorSE3 {
            w: w,
            v: OVector::<f64, U3>::new(5.6, 6.7, 7.8),
        };
        let hat4 = vec6.hat();
        let group = LieGroupSE3::from_hat(&hat4);
        let hat4_back = group.to_hat().unwrap();
        let vec6_back = LieVectorSE3::from_hat(&hat4_back);
        assert!((&vec6_back.w - &vec6.w).norm() < 1e-10);
        assert!((&vec6_back.v - &vec6.v).norm() < 1e-10);
        assert!((&hat4_back - hat4).norm() < 1e-10);
    }
    #[test]
    fn test_adjoint_matrix() {
        let w = OVector::<f64, U3>::new(1.2, 2.5, 3.7);
        let w = w / w.norm();
        let vec6 = LieVectorSE3 {
            w: w,
            v: OVector::<f64, U3>::new(5.6, 6.7, 7.8),
        };
        let group = LieGroupSE3::from(&vec6);

        let tau = OVector::<f64, U3>::new(5.2, 5.5, 2.7);
        let tau = w / w.norm();
        let tau_vec = LieVectorSE3 {
            w: w,
            v: OVector::<f64, U3>::new(1.6, 2.7, 3.8),
        };
        let tau_hat = tau_vec.hat();

        // 1. use adjoint_action
        let tau_hat_i = group.adjoint_action(&tau_hat).unwrap();
        let tau_vec_i = LieVectorSE3::from_hat(&tau_hat_i);

        // 2. use adjoint matrix
        let adj = group.adjoint_matrix();
        let tau_column = tau_vec.to_column_vector();
        let tau_column_j = adj * tau_column;
        let tau_vec_j = LieVectorSE3::from_column_vector(&tau_column_j);
        let tau_hat_j = tau_vec_j.hat();

        assert!((&tau_hat_i - &tau_hat_j).norm() < 1e-10);
        assert!((&tau_vec_i.w - &tau_vec_j.w).norm() < 1e-10);
        assert!((&tau_vec_i.v - &tau_vec_j.v).norm() < 1e-10);
    }
}
