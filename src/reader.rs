use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::repeat;
use std::marker;

use bytemuck::pod_collect_to_vec;
use heed::types::ByteSlice;
use heed::{Database, RoTxn};
use ordered_float::OrderedFloat;

use crate::node::{Descendants, Leaf, SplitPlaneNormal};
use crate::{Distance, ItemId, Node, NodeCodec, NodeId, Side, BEU32};

// TODO use a "metadata" key to store and check the dimensions and distance type
pub struct Reader<D: Distance> {
    database: heed::Database<BEU32, NodeCodec<D>>,
    roots: Vec<NodeId>,
    dimensions: usize,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance + 'static> Reader<D> {
    pub fn open<U>(
        rtxn: &RoTxn,
        database: Database<BEU32, U>,
        dimensions: usize,
    ) -> heed::Result<Reader<D>> {
        let roots = match database.remap_data_type::<ByteSlice>().get(rtxn, &u32::MAX)? {
            Some(roots_bytes) => pod_collect_to_vec(roots_bytes),
            None => Vec::new(),
        };

        Ok(Reader {
            database: database.remap_data_type(),
            roots,
            dimensions,
            _marker: marker::PhantomData,
        })
    }

    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> heed::Result<Option<Vec<f32>>> {
        Ok(item_leaf(self.database, rtxn, item)?.map(|leaf| leaf.vector.into_owned()))
    }

    pub fn nns_by_item(
        &self,
        rtxn: &RoTxn,
        item: ItemId,
        count: usize,
        search_k: Option<usize>, // TODO consider Option<NonZeroUsize>
    ) -> heed::Result<Option<Vec<(ItemId, f32)>>> {
        match item_leaf(self.database, rtxn, item)? {
            Some(leaf) => self.nns_by_leaf(rtxn, &leaf, count, search_k).map(Some),
            None => Ok(None),
        }
    }

    pub fn nns_by_vector(
        &self,
        rtxn: &RoTxn,
        vector: &[f32],
        count: usize,
        search_k: Option<usize>, // TODO consider Option<NonZeroUsize>
    ) -> heed::Result<Vec<(ItemId, f32)>> {
        assert_eq!(
            vector.len(),
            self.dimensions,
            "invalid vector dimensions, provided {} but expected {}",
            vector.len(),
            self.dimensions
        );

        let leaf = Leaf { header: D::new_header(vector), vector: Cow::Borrowed(vector) };
        self.nns_by_leaf(rtxn, &leaf, count, search_k)
    }

    fn nns_by_leaf(
        &self,
        rtxn: &RoTxn,
        query_leaf: &Leaf<D>,
        count: usize,
        search_k: Option<usize>, // TODO consider Option<NonZeroUsize>
    ) -> heed::Result<Vec<(ItemId, f32)>> {
        // TODO define the capacity
        let mut queue = BinaryHeap::new();
        let search_k = search_k.unwrap_or(count * self.roots.len());

        // Insert all the root nodes and associate them to the highest distance.
        queue.extend(repeat(OrderedFloat(f32::INFINITY)).zip(self.roots.iter().copied()));

        let mut nns = Vec::<ItemId>::new();
        while nns.len() < search_k {
            let (OrderedFloat(dist), item) = match queue.pop() {
                Some(out) => out,
                None => break,
            };

            match self.database.get(rtxn, &item)?.unwrap() {
                Node::Leaf(_) => nns.push(item),
                // TODO introduce an iterator method to avoid deserializing into a vector
                Node::Descendants(Descendants { mut descendants }) => match descendants {
                    Cow::Borrowed(descendants) => nns.extend_from_slice(descendants),
                    Cow::Owned(ref mut descendants) => nns.append(descendants),
                },
                Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                    let margin = D::margin(&normal, &query_leaf.vector);
                    queue.push((OrderedFloat(D::pq_distance(dist, margin, Side::Left)), left));
                    queue.push((OrderedFloat(D::pq_distance(dist, margin, Side::Right)), right));
                }
            }
        }

        // Get distances for all items
        // To avoid calculating distance multiple times for any items, sort by id and dedup by id.
        nns.sort_unstable();
        nns.dedup();

        let mut nns_distances = Vec::with_capacity(nns.len());
        for nn in nns {
            let leaf = match self.database.get(rtxn, &nn)?.unwrap() {
                Node::Leaf(leaf) => leaf,
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => panic!("Shouldn't happen"),
            };
            let distance = D::distance(query_leaf, &leaf);
            nns_distances.push(Reverse((OrderedFloat(distance), nn)));
        }

        let mut sorted_nns = BinaryHeap::from(nns_distances);
        let capacity = count.min(sorted_nns.len());
        let mut output = Vec::with_capacity(capacity);
        while let Some(Reverse((OrderedFloat(dist), item))) = sorted_nns.pop() {
            if output.len() == capacity {
                break;
            }
            output.push((item, D::normalized_distance(dbg!(dist))));
        }

        Ok(output)
    }
}

pub fn item_leaf<'a, D: Distance + 'static>(
    database: Database<BEU32, NodeCodec<D>>,
    rtxn: &'a RoTxn,
    item: ItemId,
) -> heed::Result<Option<Leaf<'a, D>>> {
    match database.get(rtxn, &item)? {
        Some(Node::Leaf(leaf)) => Ok(Some(leaf)),
        Some(Node::SplitPlaneNormal(_)) => Ok(None),
        Some(Node::Descendants(_)) => Ok(None),
        None => Ok(None),
    }
}

#[cfg(test)]
mod test {
    use std::fmt::Display;

    use super::*;
    use crate::writer::test::*;
    use crate::{Angular, Writer};

    pub struct NnsRes(pub Option<Vec<(ItemId, f32)>>);

    impl Display for NnsRes {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self.0 {
                Some(ref vec) => {
                    for (id, dist) in vec {
                        writeln!(f, "id({id}): distance({dist})")?;
                    }
                    Ok(())
                }
                None => f.write_str("No results found"),
            }
        }
    }

    #[test]
    fn two_db_with_wrong_dimension() {
        let handle = create_database();
        let mut wtxn = handle.env.write_txn().unwrap();
        let writer = Writer::<Angular>::prepare(&mut wtxn, 2, handle.database).unwrap();
        writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();

        writer.build(&mut wtxn, rng(), Some(1)).unwrap();
        wtxn.commit().unwrap();

        let rtxn = handle.env.read_txn().unwrap();
        // TODO: Should get an error
        let reader = Reader::<Angular>::open(&rtxn, handle.database, 4).unwrap();
        let ret = reader.nns_by_item(&rtxn, 0, 5, None).unwrap();

        insta::assert_display_snapshot!(NnsRes(ret), @r###"
        id(0): distance(1.4142135)
        "###);
    }

    #[test]
    fn two_dimension_on_a_line() {
        let handle = create_database();
        let mut wtxn = handle.env.write_txn().unwrap();
        let writer = Writer::<Angular>::prepare(&mut wtxn, 2, handle.database).unwrap();
        // We'll draw a simple line over the y as seen below
        // (0,0) # . . . . .
        // (0,1) # . . . . .
        // (0,2) # . . . . .
        // (0,3) # . . . . .
        // [...]
        for i in 0..100 {
            writer.add_item(&mut wtxn, i, &[0.0, i as f32]).unwrap();
        }

        writer.build(&mut wtxn, rng(), Some(50)).unwrap();
        wtxn.commit().unwrap();

        let rtxn = handle.env.read_txn().unwrap();
        let reader = Reader::<Angular>::open(&rtxn, handle.database, 2).unwrap();

        // if we can't look into any node we can't find anything
        let ret = reader.nns_by_item(&rtxn, 0, 5, Some(0)).unwrap();
        insta::assert_display_snapshot!(NnsRes(ret), @"");

        // if we can't look into enough nodes we find some random points
        let ret = reader.nns_by_item(&rtxn, 0, 5, Some(1)).unwrap();
        // TODO: The distances are wrong
        insta::assert_display_snapshot!(NnsRes(ret), @r###"
        id(9): distance(1.4142135)
        id(70): distance(1.4142135)
        "###);

        // if we can look into all the node there is no inifinite loop and it works
        let ret = reader.nns_by_item(&rtxn, 0, 5, Some(usize::MAX)).unwrap();
        // TODO: The distances are wrong
        insta::assert_display_snapshot!(NnsRes(ret), @r###"
        id(0): distance(1.4142135)
        id(1): distance(1.4142135)
        id(2): distance(1.4142135)
        id(3): distance(1.4142135)
        id(4): distance(1.4142135)
        "###);

        let ret = reader.nns_by_item(&rtxn, 0, 5, None).unwrap();
        // TODO: The distances are wrong
        insta::assert_display_snapshot!(NnsRes(ret), @r###"
        id(0): distance(1.4142135)
        id(1): distance(1.4142135)
        id(2): distance(1.4142135)
        id(3): distance(1.4142135)
        id(4): distance(1.4142135)
        "###);
    }

    #[test]
    fn two_dimension_on_a_column() {
        let handle = create_database();
        let mut wtxn = handle.env.write_txn().unwrap();
        let writer = Writer::<Angular>::prepare(&mut wtxn, 2, handle.database).unwrap();
        // We'll draw a simple line over the y as seen below
        // (0,0) # # # # # # ...
        for i in 0..100 {
            writer.add_item(&mut wtxn, i, &[i as f32, 0.0]).unwrap();
        }

        writer.build(&mut wtxn, rng(), Some(50)).unwrap();
        wtxn.commit().unwrap();

        let rtxn = handle.env.read_txn().unwrap();
        let reader = Reader::<Angular>::open(&rtxn, handle.database, 2).unwrap();
        let ret = reader.nns_by_item(&rtxn, 0, 5, None).unwrap();

        // TODO: The distances are wrong
        insta::assert_display_snapshot!(NnsRes(ret), @r###"
        id(0): distance(1.4142135)
        id(1): distance(1.4142135)
        id(2): distance(1.4142135)
        id(3): distance(1.4142135)
        id(4): distance(1.4142135)
        "###);
    }
}
