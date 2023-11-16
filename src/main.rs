// use annoy_rs::*;
// use arroy::{ArroyReader, DistanceType, HeedReader};
// use heed::EnvOpenOptions;

// const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024;

// fn main() -> heed::Result<()> {
//     let dimensions = 40;
//     let distance_type = DistanceType::Angular;
//     let tree = std::fs::read("test.tree").unwrap();
//     // let arroy = ArroyReader::new(&tree[..], dimensions, distance_type);

//     let dir = tempfile::tempdir()?;
//     let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

//     // we will open the default unnamed database
//     let mut wtxn = env.write_txn()?;
//     let database = env.create_database(&mut wtxn, None)?;
//     HeedReader::load_from_tree(&mut wtxn, database, dimensions, distance_type, &tree)?;
//     wtxn.commit()?;

//     let rtxn = env.read_txn()?;
//     let arroy = HeedReader::new(&rtxn, database, dimensions, distance_type)?;
//     // dbg!(&arroy);
//     let v = arroy.item_vector(&rtxn, 0)?.unwrap();
//     let results = arroy.nns_by_item(&rtxn, 0, 3, None)?.unwrap();
//     println!("{v:?}");

//     let arroy = ArroyReader::new(&tree[..], dimensions, distance_type);
//     // dbg!(&arroy);
//     let classic_v = arroy.item_vector(0).unwrap();
//     let classic_results = arroy.nns_by_item(0, 3, None).unwrap();

//     // let index = AnnoyIndex::load(40, "test.tree", IndexType::Angular).unwrap();
//     // // dbg!(&index);
//     // let v0 = index.get_item_vector(0);
//     // let results0 = index.get_nearest_to_item(0, 3, -1, true);
//     // println!("{v0:?}");

//     assert_eq!(v, classic_v);

//     assert_eq!(results[0].0, classic_results[0].0 as u32);
//     assert_eq!(results[1].0, classic_results[1].0 as u32);
//     assert_eq!(results[2].0, classic_results[2].0 as u32);

//     assert_eq!(results[0].1, classic_results[0].1);
//     assert_eq!(results[1].1, classic_results[1].1);
//     assert_eq!(results[2].1, classic_results[2].1);

//     Ok(())
// }

use arroy::{Angular, Euclidean, NodeCodec, Reader, Writer, BEU32};
use bytemuck::pod_collect_to_vec;
use heed::types::{ByteSlice, LazyDecode};
use heed::{Database, EnvOpenOptions, Unspecified};

const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024;

fn main() -> heed::Result<()> {
    let dir = tempfile::tempdir()?;
    let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

    // we will open the default unnamed database
    let mut wtxn = env.write_txn()?;
    let dimensions = 2;
    let database: Database<BEU32, Unspecified> = env.create_database(&mut wtxn, None)?;
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, dimensions, database)?;

    for i in 0..5 {
        let f = i as f32;
        writer.add_item(&mut wtxn, i, &[0.0, f])?;
    }

    let rng = rand::thread_rng();
    writer.build(&mut wtxn, rng, Some(1))?;

    for result in database.remap_data_type::<LazyDecode<NodeCodec<Euclidean>>>().iter(&wtxn)? {
        let (i, lazy_node) = result?;
        if i != u32::MAX {
            let node = lazy_node.decode().unwrap();
            println!("{i}: {node:?}");
        } else {
            let roots_bytes = database.remap_data_type::<ByteSlice>().get(&wtxn, &i)?.unwrap();
            let roots: Vec<u32> = pod_collect_to_vec(roots_bytes);
            println!("u32::MAX: {roots:?}");
        }
    }
    wtxn.commit()?;

    let rtxn = env.read_txn()?;
    let reader = Reader::<Euclidean>::open(&rtxn, database, dimensions)?;
    for (id, dist) in reader.nns_by_vector(&rtxn, &[0., 0.], 10, None)? {
        println!("id({id}): distance({dist})");
    }

    // HeedReader::load_from_tree(&mut wtxn, database, dimensions, distance_type, &tree)?;

    Ok(())
}
