use std::fmt::Display;
use std::num::NonZeroUsize;

use super::*;
use crate::distances::{Euclidean, Manhattan};
use crate::{ItemId, Reader, Writer};

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
fn open_unfinished_db() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 2).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();
    wtxn.commit().unwrap();

    let rtxn = handle.env.read_txn().unwrap();
    let ret = Reader::<Euclidean>::open(&rtxn, 0, handle.database).map(|_| ()).unwrap_err();
    insta::assert_display_snapshot!(ret, @"Metadata are missing, did you build your database before trying to read it.");
}

#[test]
fn open_db_with_wrong_dimension() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 2).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();

    writer.build(&mut wtxn, rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    let rtxn = handle.env.read_txn().unwrap();
    let reader = Reader::<Euclidean>::open(&rtxn, 0, handle.database).unwrap();
    let ret = reader.nns_by_vector(&rtxn, &[1.0, 2.0, 3.0], 5, None).unwrap_err();
    insta::assert_display_snapshot!(ret, @"Invalid vector dimensions. Got 3 but expected 2.");
}

#[test]
#[should_panic]
fn open_db_with_wrong_distance() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 2).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();

    writer.build(&mut wtxn, rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    let rtxn = handle.env.read_txn().unwrap();
    let reader = Reader::<Manhattan>::open(&rtxn, 0, handle.database).unwrap();
    reader.nns_by_vector(&rtxn, &[1.0, 2.0], 5, None).unwrap();
}

#[test]
fn two_dimension_on_a_line() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 2).unwrap();
    // We'll draw a simple line over the y as seen below
    // (0,0) # # # # # # ...
    for i in 0..100 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.0]).unwrap();
    }

    writer.build(&mut wtxn, rng(), Some(50)).unwrap();
    wtxn.commit().unwrap();

    let rtxn = handle.env.read_txn().unwrap();
    let reader = Reader::<Euclidean>::open(&rtxn, 0, handle.database).unwrap();

    // if we can't look into enough nodes we find some random points
    let ret = reader.nns_by_item(&rtxn, 0, 5, NonZeroUsize::new(1)).unwrap();
    insta::assert_display_snapshot!(NnsRes(ret), @r###"
    id(33): distance(33)
    id(69): distance(69)
    "###);

    // if we can look into all the node there is no inifinite loop and it works
    let ret = reader.nns_by_item(&rtxn, 0, 5, NonZeroUsize::new(usize::MAX)).unwrap();
    insta::assert_display_snapshot!(NnsRes(ret), @r###"
    id(0): distance(0)
    id(1): distance(1)
    id(2): distance(2)
    id(3): distance(3)
    id(4): distance(4)
    "###);

    let ret = reader.nns_by_item(&rtxn, 0, 5, None).unwrap();
    insta::assert_display_snapshot!(NnsRes(ret), @r###"
    id(1): distance(1)
    id(2): distance(2)
    id(3): distance(3)
    id(4): distance(4)
    id(5): distance(5)
    "###);
}

#[test]
fn two_dimension_on_a_column() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 2).unwrap();
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
    let reader = Reader::<Euclidean>::open(&rtxn, 0, handle.database).unwrap();
    let ret = reader.nns_by_item(&rtxn, 0, 5, None).unwrap();

    insta::assert_display_snapshot!(NnsRes(ret), @r###"
    id(1): distance(1)
    id(2): distance(2)
    id(3): distance(3)
    id(4): distance(4)
    id(5): distance(5)
    "###);
}
