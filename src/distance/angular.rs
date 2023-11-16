use bytemuck::{Pod, Zeroable};
use rand::seq::SliceRandom;
use rand::Rng;

use super::{cosine_distance_no_simd, dot_product_no_simd};
use crate::node::{Leaf, SplitPlaneNormal};
use crate::{Distance, Side};

#[derive(Debug, Clone)]
pub enum Angular {}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderAngular {
    norm: f32,
}

impl Distance for Angular {
    type Header = NodeHeaderAngular;

    fn new_header(vector: &[f32]) -> Self::Header {
        NodeHeaderAngular { norm: Self::norm(vector) }
    }

    fn distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        cosine_distance_no_simd(&p.vector, &q.vector)
    }

    fn normalized_distance(d: f32) -> f32 {
        d.max(0.0).sqrt()
    }

    fn norm(v: &[f32]) -> f32 {
        dot_product_no_simd(v, v).sqrt()
    }

    fn normalize(node: &mut Leaf<Self>) {
        let norm = Self::norm(&node.vector);
        if norm > 0.0 {
            node.vector.iter_mut().for_each(|x| *x /= norm);
        }
    }

    fn init(node: &mut Leaf<Self>) {
        node.header.norm = dot_product_no_simd(&node.vector, &node.vector);
    }

    fn update_mean(mean: &mut Leaf<Self>, new_node: &Leaf<Self>, norm: f32, c: f32) {
        mean.vector
            .iter_mut()
            .zip(&new_node.vector)
            .for_each(|(x, n)| *x = (*x * c + *n / norm) / (c + 1.0));
    }

    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> SplitPlaneNormal {
        let [node_p, node_q] = two_means(rng, children, true);
        let vector = node_p.vector.iter().zip(node_q.vector.iter()).map(|(&p, &q)| p - q).collect();
        let mut normal = Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector };
        Self::normalize(&mut normal);
        // TODO we are returning invalid left and rights
        SplitPlaneNormal { normal: normal.vector, left: u32::MAX, right: u32::MAX }
    }

    fn margin(p: &[f32], q: &[f32]) -> f32 {
        dot_product_no_simd(p, q)
    }

    fn side<R: Rng>(plane: &SplitPlaneNormal, node: &Leaf<Self>, rng: &mut R) -> Side {
        let dot = Self::margin(&node.vector, &plane.normal);
        if dot > 0.0 {
            Side::Right
        } else if dot < 0.0 {
            Side::Left
        } else {
            Side::random(rng)
        }
    }
}

fn two_means<D: Distance, R: Rng>(rng: &mut R, leafs: &[Leaf<D>], cosine: bool) -> [Leaf<D>; 2] {
    // This algorithm is a huge heuristic. Empirically it works really well, but I
    // can't motivate it well. The basic idea is to keep two centroids and assign
    // points to either one of them. We weight each centroid by the number of points
    // assigned to it, so to balance it.

    const ITERATION_STEPS: usize = 200;

    let mut random_nodes = leafs.choose_multiple(rng, 2);
    let mut leaf_p = random_nodes.next().unwrap().clone();
    let mut leaf_q = random_nodes.next().unwrap().clone();

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
