use arroy::{DotProduct, KeyCodec, NodeCodec, NodeMode, Reader, Result, Writer};
use heed::types::LazyDecode;
use heed::{DatabaseFlags, EnvOpenOptions, Unspecified};

const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024;

fn main() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

    // we will open the default unnamed database
    let mut wtxn = env.write_txn()?;
    let dimensions = 2;
    let database = env
        .database_options()
        .types::<KeyCodec, Unspecified>()
        .flags(DatabaseFlags::INTEGER_KEY)
        .create(&mut wtxn)?;
    let writer = Writer::<DotProduct>::prepare(&mut wtxn, database, 0, dimensions)?;

    for i in 0..5 {
        let f = i as f32;
        writer.add_item(&mut wtxn, i, &[1.0, f])?;
    }

    let rng = rand::thread_rng();
    writer.build(&mut wtxn, rng, Some(1))?;

    for result in database.remap_data_type::<LazyDecode<NodeCodec<DotProduct>>>().iter(&wtxn)? {
        let (key, lazy_node) = result?;
        if key.node.mode != NodeMode::Metadata {
            let node = lazy_node.decode().unwrap();
            println!("{}: {node:?}", key.node.item);
        }
    }
    wtxn.commit()?;

    let rtxn = env.read_txn()?;
    let reader = Reader::<DotProduct>::open(&rtxn, 0, database)?;
    for (id, dist) in reader.nns_by_item(&rtxn, 0, 10, None)?.unwrap() {
        println!("id({id}): distance({dist})");
    }

    // HeedReader::load_from_tree(&mut wtxn, database, dimensions, distance_type, &tree)?;

    Ok(())
}
