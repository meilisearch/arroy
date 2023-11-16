use std::fmt;

pub use angular::Angular;
use bytemuck::{Pod, Zeroable};
pub use euclidean::Euclidean;
pub use manhattan::Manhattan;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::node::{Leaf, SplitPlaneNormal};
use crate::Side;

mod angular;
mod euclidean;
mod manhattan;

pub trait Distance: Sized + Clone + fmt::Debug {
    type Header: Pod + Zeroable + fmt::Debug;

    fn new_header(vector: &[f32]) -> Self::Header;

    /// Returns a non-normalized distance.
    fn distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32;

    /// Normalizes the distance returned by the distance method.
    fn normalized_distance(d: f32) -> f32 {
        d.sqrt()
    }

    fn pq_distance(distance: f32, margin: f32, side: Side) -> f32 {
        match side {
            Side::Left => (-margin).min(distance),
            Side::Right => margin.min(distance),
        }
    }

    fn norm(v: &[f32]) -> f32 {
        dot_product_no_simd(v, v).sqrt()
    }

    fn normalize(node: &mut Leaf<Self>) {
        let norm = Self::norm(&node.vector);
        if norm > 0.0 {
            node.vector.to_mut().iter_mut().for_each(|x| *x /= norm);
        }
    }

    fn init(node: &mut Leaf<Self>);

    fn update_mean(mean: &mut Leaf<Self>, new_node: &Leaf<Self>, norm: f32, c: f32) {
        mean.vector
            .to_mut()
            .iter_mut()
            .zip(new_node.vector.iter())
            .for_each(|(x, n)| *x = (*x * c + *n / norm) / (c + 1.0));
    }

    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> SplitPlaneNormal<'static>;

    fn margin(p: &Leaf<Self>, q: &[f32]) -> f32 {
        Self::margin_no_header(&p.vector, q)
    }

    fn margin_no_header(p: &[f32], q: &[f32]) -> f32;

    fn side<R: Rng>(plane: &SplitPlaneNormal, node: &Leaf<Self>, rng: &mut R) -> Side {
        let dot = Self::margin(node, &plane.normal);
        if dot > 0.0 {
            Side::Right
        } else if dot < 0.0 {
            Side::Left
        } else {
            Side::random(rng)
        }
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderDot {
    dot_factor: f32,
}

fn dot_product_no_simd(u: &[f32], v: &[f32]) -> f32 {
    u.iter().zip(v.iter()).map(|(x, y)| x * y).sum()
}

fn minkowski_margin(u: &[f32], v: &[f32], bias: f32) -> f32 {
    bias + dot_product_no_simd(u, v)
}

fn cosine_distance_no_simd(u: &[f32], v: &[f32]) -> f32 {
    // want to calculate (a/|a| - b/|b|)^2
    // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
    // = 2 - 2cos
    let mut pp: f32 = 0.0;
    let mut qq: f32 = 0.0;
    let mut pq: f32 = 0.0;
    for (u, v) in u.iter().zip(v.iter()) {
        pp += u * u;
        qq += v * v;
        pq += u * v;
    }
    let ppqq = dbg!(pp) * dbg!(qq);
    if ppqq.is_sign_positive() {
        2.0 - dbg!(2.0 * dbg!(pq)) / dbg!(ppqq.sqrt())
    } else {
        2.0
    }
}

fn manhattan_distance_no_simd(u: &[f32], v: &[f32]) -> f32 {
    u.iter().zip(v.iter()).map(|(x, y)| (x - y).abs()).sum()
}

fn euclidean_distance_no_simd(u: &[f32], v: &[f32]) -> f32 {
    u.iter().zip(v.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn two_means<D: Distance, R: Rng>(
    rng: &mut R,
    leafs: &[Leaf<D>],
    cosine: bool,
) -> [Leaf<'static, D>; 2] {
    // This algorithm is a huge heuristic. Empirically it works really well, but I
    // can't motivate it well. The basic idea is to keep two centroids and assign
    // points to either one of them. We weight each centroid by the number of points
    // assigned to it, so to balance it.

    const ITERATION_STEPS: usize = 200;

    let mut random_nodes = leafs.choose_multiple(rng, 2);
    let mut leaf_p = random_nodes.next().unwrap().clone().into_owned();
    let mut leaf_q = random_nodes.next().unwrap().clone().into_owned();

    if cosine {
        D::normalize(&mut leaf_p);
        D::normalize(&mut leaf_q);
    }

    D::init(&mut leaf_p);
    D::init(&mut leaf_q);

    let mut ic = 1.0;
    let mut jc = 1.0;
    for _ in 0..ITERATION_STEPS {
        let node_k = leafs.choose(rng).unwrap();
        let di = ic * D::distance(&leaf_p, node_k);
        let dj = jc * D::distance(&leaf_q, node_k);
        let norm = if cosine { D::norm(&node_k.vector) } else { 1.0 };
        if norm.is_nan() || norm <= 0.0 {
            continue;
        }
        if di < dj {
            Distance::update_mean(&mut leaf_p, node_k, norm, ic);
            Distance::init(&mut leaf_p);
            ic += 1.0;
        } else if dj < di {
            Distance::update_mean(&mut leaf_q, node_k, norm, jc);
            Distance::init(&mut leaf_q);
            jc += 1.0;
        }
    }

    [leaf_p, leaf_q]
}
