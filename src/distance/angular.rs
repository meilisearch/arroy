use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::two_means;
use crate::node::{Leaf, UnalignedF32Slice};
use crate::spaces::simple::dot_product;
use crate::Distance;

#[derive(Debug, Clone)]
pub enum Angular {}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderAngular {
    norm: f32,
}

impl Distance for Angular {
    type Header = NodeHeaderAngular;

    fn new_header(vector: &UnalignedF32Slice) -> Self::Header {
        NodeHeaderAngular { norm: Self::norm_no_header(vector) }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        let pp = p.header.norm;
        let qq = q.header.norm;
        let pq = dot_product(&p.vector, &q.vector);
        let ppqq = pp * qq;
        if ppqq >= f32::MIN_POSITIVE {
            2.0 - 2.0 * pq / ppqq.sqrt()
        } else {
            2.0 // cos is 0
        }
    }

    fn init(node: &mut Leaf<Self>) {
        node.header.norm = dot_product(&node.vector, &node.vector);
    }

    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> Vec<f32> {
        let [node_p, node_q] = two_means(rng, children, true);
        let vector = node_p.vector.iter().zip(node_q.vector.iter()).map(|(p, q)| p - q).collect();
        let mut normal =
            Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: Cow::Owned(vector) };
        Self::normalize(&mut normal);

        normal.vector.into_owned()
    }

    fn margin_no_header(p: &UnalignedF32Slice, q: &UnalignedF32Slice) -> f32 {
        dot_product(p, q)
    }
}
