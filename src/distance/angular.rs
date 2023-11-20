use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::two_means;
use crate::node::{Leaf, SplitPlaneNormal};
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

    fn new_header(vector: &[f32]) -> Self::Header {
        NodeHeaderAngular { norm: Self::norm(vector) }
    }

    fn distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
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

    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> SplitPlaneNormal<'static> {
        let [node_p, node_q] = two_means(rng, children, true);
        let vector = node_p.vector.iter().zip(node_q.vector.iter()).map(|(&p, &q)| p - q).collect();
        let mut normal = Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector };
        Self::normalize(&mut normal);
        // TODO we are returning invalid left and rights
        SplitPlaneNormal { normal: normal.vector, left: u32::MAX, right: u32::MAX }
    }

    fn margin_no_header(p: &[f32], q: &[f32]) -> f32 {
        dot_product(p, q)
    }
}
