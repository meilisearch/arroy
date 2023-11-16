use std::fmt;

pub use angular::Angular;
use bytemuck::{Pod, Zeroable};
use rand::Rng;

use crate::node::{Leaf, SplitPlaneNormal};
use crate::Side;

mod angular;

pub trait Distance: Sized + Clone + fmt::Debug {
    type Header: Pod + Zeroable + fmt::Debug;

    fn new_header(vector: &[f32]) -> Self::Header;
    /// Returns a non-normalized distance.
    fn distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32;
    /// Normalizes the distance returned by the distance method.
    fn normalized_distance(d: f32) -> f32;
    fn pq_distance(distance: f32, margin: f32, side: Side) -> f32;
    fn norm(v: &[f32]) -> f32;
    fn normalize(node: &mut Leaf<Self>);
    fn init(node: &mut Leaf<Self>);
    fn update_mean(mean: &mut Leaf<Self>, new_node: &Leaf<Self>, norm: f32, c: f32);
    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> SplitPlaneNormal;
    fn margin(p: &[f32], q: &[f32]) -> f32;
    fn side<R: Rng>(plane: &SplitPlaneNormal, node: &Leaf<Self>, rng: &mut R) -> Side;
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderMinkowski {
    bias: f32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderDot {
    dot_factor: f32,
}

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
