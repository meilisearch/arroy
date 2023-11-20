use std::fmt::Display;
use std::num::NonZeroUsize;

use super::*;
use crate::{Angular, ItemId, Reader, Writer};

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
    let writer = Writer::<Angular>::prepare(&mut wtxn, handle.database, 2).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();

    writer.build(&mut wtxn, rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    let rtxn = handle.env.read_txn().unwrap();
    let reader = Reader::<Angular>::open(&rtxn, handle.database).unwrap();
    let ret = reader.nns_by_item(&rtxn, 0, 5, None).unwrap();

    insta::assert_display_snapshot!(NnsRes(ret), @r###"
    id(0): distance(1.4142135)
    "###);
}

#[test]
fn two_dimension_on_a_line() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Angular>::prepare(&mut wtxn, handle.database, 2).unwrap();
    // We'll draw a simple line over the y as seen below
    // (0,0) # # # # # # ...
    for i in 0..100 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.0]).unwrap();
    }

    writer.build(&mut wtxn, rng(), Some(50)).unwrap();
    wtxn.commit().unwrap();

    let rtxn = handle.env.read_txn().unwrap();
    let reader = Reader::<Angular>::open(&rtxn, handle.database).unwrap();

    // if we can't look into enough nodes we find some random points
    let ret = reader.nns_by_item(&rtxn, 0, 5, NonZeroUsize::new(1)).unwrap();
    insta::assert_display_snapshot!(NnsRes(ret), @r###"
    id(9): distance(1.4142135)
    id(70): distance(1.4142135)
    "###);

    // if we can look into all the node there is no inifinite loop and it works
    let ret = reader.nns_by_item(&rtxn, 0, 5, NonZeroUsize::new(usize::MAX)).unwrap();
    insta::assert_display_snapshot!(NnsRes(ret), @r###"
    id(0): distance(1.4142135)
    id(1): distance(1.4142135)
    id(2): distance(1.4142135)
    id(3): distance(1.4142135)
    id(4): distance(1.4142135)
    "###);

    let ret = reader.nns_by_item(&rtxn, 0, 5, None).unwrap();
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
    let writer = Writer::<Angular>::prepare(&mut wtxn, handle.database, 2).unwrap();
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
    let reader = Reader::<Angular>::open(&rtxn, handle.database).unwrap();
    let ret = reader.nns_by_item(&rtxn, 0, 5, None).unwrap();

    insta::assert_display_snapshot!(NnsRes(ret), @r###"
    id(0): distance(1.4142135)
    id(1): distance(1.4142135)
    id(2): distance(1.4142135)
    id(3): distance(1.4142135)
    id(4): distance(1.4142135)
    "###);
}
