use arroy::{Euclidean, NodeCodec, Reader, Result, Writer, BEU32};
use bytemuck::pod_collect_to_vec;
use heed::types::{ByteSlice, LazyDecode};
use heed::{Database, EnvOpenOptions, Unspecified};

const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024;

fn main() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

    // we will open the default unnamed database
    let mut wtxn = env.write_txn()?;
    let dimensions = 2;
    let database: Database<BEU32, Unspecified> = env.create_database(&mut wtxn, None)?;
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, database, dimensions)?;

    for i in 0..5 {
        let f = i as f32;
        writer.add_item(&mut wtxn, i, &[f, f])?;
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
    let reader = Reader::<Euclidean>::open(&rtxn, database)?;
    for (id, dist) in reader.nns_by_vector(&rtxn, &[0., 0.], 10, None)? {
        println!("id({id}): distance({dist})");
    }

    // HeedReader::load_from_tree(&mut wtxn, database, dimensions, distance_type, &tree)?;

    Ok(())
}
