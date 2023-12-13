use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::two_means;
use crate::distance::Distance;
use crate::node::{Leaf, UnalignedF32Slice};
use crate::parallel::ImmutableSubsetLeafs;
use crate::spaces::simple::dot_product;

/// The Cosine similarity is a measure of similarity between two
/// non-zero vectors defined in an inner product space. Cosine similarity
/// is the cosine of the angle between the vectors.
#[derive(Debug, Clone)]
pub enum Angular {}

/// The header of Angular leaf nodes.
#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderAngular {
    norm: f32,
}

impl Distance for Angular {
    type Header = NodeHeaderAngular;

    fn name() -> &'static str {
        "angular"
    }

    fn new_header(vector: &UnalignedF32Slice) -> Self::Header {
        NodeHeaderAngular { norm: Self::norm_no_header(vector) }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        let pn = p.header.norm;
        let qn = q.header.norm;
        let pq = dot_product(&p.vector, &q.vector);
        let pnqn = pn * qn;
        let cos = pq / pnqn;

        // cos is [-1; 1]
        // cos =  0. -> 0.5
        // cos = -1. -> 1.0
        // cos =  1. -> 0.0
        (1.0 - cos) / 2.0
    }

    fn normalized_distance(d: f32) -> f32 {
        d
    }

    fn init(node: &mut Leaf<Self>) {
        node.header.norm = dot_product(&node.vector, &node.vector).sqrt();
    }

    fn create_split<R: Rng>(
        children: &ImmutableSubsetLeafs<Self>,
        rng: &mut R,
    ) -> heed::Result<Vec<f32>> {
        let [node_p, node_q] = two_means(rng, children, true)?;
        let vector = node_p.vector.iter().zip(node_q.vector.iter()).map(|(p, q)| p - q).collect();
        let mut normal =
            Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: Cow::Owned(vector) };
        Self::normalize(&mut normal);

        Ok(normal.vector.into_owned())
    }

    fn margin_no_header(p: &UnalignedF32Slice, q: &UnalignedF32Slice) -> f32 {
        dot_product(p, q)
    }
}
