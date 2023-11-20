use arroy::{DotProduct, NodeCodec, Reader, Result, Writer, BEU32};
use heed::types::LazyDecode;
use heed::{Database, EnvOpenOptions, Unspecified};

const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024;

fn main() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

    // we will open the default unnamed database
    let mut wtxn = env.write_txn()?;
    let dimensions = 2;
    let database: Database<BEU32, Unspecified> = env.create_database(&mut wtxn, None)?;
    let writer = Writer::<DotProduct>::prepare(&mut wtxn, database, dimensions)?;

    for i in 0..5 {
        let f = i as f32;
        writer.add_item(&mut wtxn, i, &[1.0, f])?;
    }

    let rng = rand::thread_rng();
    writer.build(&mut wtxn, rng, Some(1))?;

    for result in database.remap_data_type::<LazyDecode<NodeCodec<DotProduct>>>().iter(&wtxn)? {
        let (i, lazy_node) = result?;
        if i != u32::MAX {
            let node = lazy_node.decode().unwrap();
            println!("{i}: {node:?}");
        }
    }
    wtxn.commit()?;

    let rtxn = env.read_txn()?;
    let reader = Reader::<DotProduct>::open(&rtxn, database)?;
    for (id, dist) in reader.nns_by_item(&rtxn, 0, 10, None)?.unwrap() {
        println!("id({id}): distance({dist})");
    }

    // HeedReader::load_from_tree(&mut wtxn, database, dimensions, distance_type, &tree)?;

    Ok(())
}
