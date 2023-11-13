pub fn dot_product_no_simd(u: &[f32], v: &[f32]) -> f32 {
    u.iter().zip(v.iter()).map(|(x, y)| x * y).sum()
}

pub fn minkowski_margin(u: &[f32], v: &[f32], bias: f32) -> f32 {
    bias + dot_product_no_simd(u, v)
}

pub fn cosine_distance_no_simd(u: &[f32], v: &[f32]) -> f32 {
    // want to calculate (a/|a| - b/|b|)^2
    // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
    // = 2 - 2cos
    let mut pp: f32 = 0.0;
    let mut qq: f32 = 0.0;
    let mut pq: f32 = 0.0;
    for (_u, _v) in u.iter().zip(v.iter()) {
        pp += _u * _u;
        qq += _v * _v;
        pq += _u * _v;
    }
    let ppqq = pp * qq;
    if ppqq.is_sign_positive() {
        2.0 - 2.0 * pq / ppqq.sqrt()
    } else {
        2.0
    }
}

pub fn manhattan_distance_no_simd(u: &[f32], v: &[f32]) -> f32 {
    u.iter().zip(v.iter()).map(|(x, y)| (x - y).abs()).sum()
}

pub fn euclidean_distance_no_simd(u: &[f32], v: &[f32]) -> f32 {
    u.iter().zip(v.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

trait Distance {
    fn preprocess();
    fn postprocess();
    fn zero_value();
    fn copy_node();
    // fn norm(leaf: &Leaf) -> f32 {
    //     let v = leaf.vector();
    //     dot_product_no_simd(&v, &v).sqrt()
    // }
    // fn normalize(leaf: &mut OwnedLeaf) {
    //     let norm = Self::norm(leaf.as_leaf());
    //     if norm > 0.0 {
    //         leaf.vector.iter_mut().for_each(|x| *x /= norm);
    //     }
    // }
    // fn update_mean(mean: &mut OwnedLeaf, new_node: &Leaf, norm: f32, c: f32) {
    //     let new_node_vector = new_node.vector();
    //     mean.vector.iter_mut().zip(&new_node_vector).for_each(|(x, y)| {
    //         *x = (*x * c + y / norm) / c + 1.0;
    //     });
    // }
}

enum Euclidean {}

impl Distance for Euclidean {}
