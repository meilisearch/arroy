use std::time::Instant;

use arroy::{DistanceType, HeedReader};
use heed::EnvOpenOptions;

const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024;

fn main() -> heed::Result<()> {
    let dimensions = 40;
    let distance_type = DistanceType::Angular;
    let tree = std::fs::read("test.tree").unwrap();

    let dir = tempfile::tempdir()?;
    let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

    // we will open the default unnamed database
    let mut wtxn = env.write_txn()?;
    let database = env.create_database(&mut wtxn, None)?;
    HeedReader::load_from_tree(&mut wtxn, database, dimensions, distance_type, &tree)?;
    wtxn.commit()?;

    let rtxn = env.read_txn()?;
    let arroy = HeedReader::new(&rtxn, database, dimensions, distance_type)?;
    let v = arroy.item_vector(&rtxn, 0)?.unwrap();
    println!("{v:?}");

    let before = Instant::now();
    let results = arroy.nns_by_item(&rtxn, 0, 30, None)?.unwrap();
    eprintln!("It took {:.02?} to find the nns", before.elapsed());
    println!("{results:?}");

    Ok(())
}
