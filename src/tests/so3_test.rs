#[cfg(test)]
mod tests {

    use crate::lie::so3::{Alg3, Grp3, One2OneMap, Vec3, SO3};

    use crate::linalg::{Const, OMatrix, OVector, U2, U3};
    /// import Re-exports of hifitime (for time) and nalgebra (for matrix)
    use crate::{
        linalg,
        time::{Duration, Epoch, Unit},
    };
    #[test]
    fn vec_to_alg() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let alg = SO3::from_vec(v).to_alg();
        let v2 = SO3::from_alg(alg).to_vec();
        assert!((v - v2).norm() < 1e-10);
    }

    #[test]
    fn vec_to_grp() {
        let vectors = [
            Vec3::new(0.0, 0.5, 0.25),
            Vec3::new(-1.0, 20.0, 33.0),
            Vec3::new(1e-10, -1e-8, 1e-9),
            10.0 * Vec3::new(1.0, 0.0, 0.0),
            100.0 * Vec3::new(0.0, 0.0, 1.0),
        ];

        for v in vectors.into_iter() {
            let grp = SO3::from_vec(v).to_grp();
            let v2 = SO3::from_grp(grp).to_vec();

            let grp2 = SO3::from_vec(v2).to_grp();
            println!("error of grp - grp2 = {}", (grp - grp2).norm());
            assert!((grp - grp2).norm() < 1e-6);
        }
    }
}
