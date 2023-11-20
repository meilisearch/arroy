use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::two_means;
use crate::node::{Leaf, SplitPlaneNormal};
use crate::spaces::simple::dot_product;
use crate::Distance;

#[derive(Debug, Clone)]
pub enum Manhattan {}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderManhattan {
    /// An extra constant term to determine the offset of the plane
    bias: f32,
}

impl Distance for Manhattan {
    type Header = NodeHeaderManhattan;

    fn new_header(_vector: &[f32]) -> Self::Header {
        NodeHeaderManhattan { bias: 0.0 }
    }

    fn name() -> &'static str {
        "manhattan"
    }

    fn distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        p.vector.iter().zip(q.vector.iter()).map(|(&p, &q)| (p - q).abs()).sum()
    }

    fn normalized_distance(d: f32) -> f32 {
        d.max(0.0)
    }

    fn init(_node: &mut Leaf<Self>) {}

    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> SplitPlaneNormal<'static> {
        let [node_p, node_q] = two_means(rng, children, false);
        let vector = node_p.vector.iter().zip(node_q.vector.iter()).map(|(&p, &q)| p - q).collect();
        let mut normal = Leaf { header: NodeHeaderManhattan { bias: 0.0 }, vector };
        Self::normalize(&mut normal);

        normal.header.bias = normal
            .vector
            .iter()
            .zip(node_p.vector.iter())
            .zip(node_q.vector.iter())
            .map(|((&n, &p), &q)| -n * (p + q) / 2.0)
            .sum();

        // TODO we are returning invalid left and rights
        SplitPlaneNormal { normal: normal.vector, left: u32::MAX, right: u32::MAX }
    }

    fn margin(p: &Leaf<Self>, q: &[f32]) -> f32 {
        p.header.bias + dot_product(&p.vector, q)
    }

    fn margin_no_header(p: &[f32], q: &[f32]) -> f32 {
        dot_product(p, q)
    }
}
