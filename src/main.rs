use annoy_rs::*;
use arroy::{DistanceType, HeedReader};
use heed::EnvOpenOptions;

const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024;

fn main() -> heed::Result<()> {
    let dimensions = 40;
    let distance_type = DistanceType::Angular;
    let tree = std::fs::read("test.tree").unwrap();
    // let arroy = ArroyReader::new(&tree[..], dimensions, distance_type);

    let dir = tempfile::tempdir()?;
    let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

    // we will open the default unnamed database
    let mut wtxn = env.write_txn()?;
    let database = env.create_database(&mut wtxn, None)?;
    HeedReader::load_from_tree(&mut wtxn, database, dimensions, distance_type, &tree)?;
    wtxn.commit()?;

    let rtxn = env.read_txn()?;
    let arroy = HeedReader::new(&rtxn, database, dimensions, distance_type)?;
    // dbg!(&arroy);
    let v = arroy.item_vector(&rtxn, 0)?.unwrap();
    let results = arroy.nns_by_item(&rtxn, 0, 3, None)?.unwrap();
    println!("{v:?}");

    let index = AnnoyIndex::load(40, "test.tree", IndexType::Angular).unwrap();
    // dbg!(&index);
    let v0 = index.get_item_vector(0);
    let results0 = index.get_nearest_to_item(0, 3, -1, true);
    println!("{v0:?}");

    assert_eq!(v, v0);

    assert_eq!(results[0].0, results0.id_list[0] as u32);
    assert_eq!(results[1].0, results0.id_list[1] as u32);
    assert_eq!(results[2].0, results0.id_list[2] as u32);

    assert_eq!(results[0].1, results0.distance_list[0]);
    assert_eq!(results[1].1, results0.distance_list[1]);
    assert_eq!(results[2].1, results0.distance_list[2]);

    Ok(())
}
