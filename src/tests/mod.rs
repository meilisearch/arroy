use std::fmt;

use bytemuck::pod_collect_to_vec;
use heed::types::{ByteSlice, LazyDecode};
use heed::{Database, Env, EnvOpenOptions, Unspecified};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tempfile::TempDir;

use crate::{Angular, NodeCodec, BEU32};

mod reader;
mod writer;

pub struct DatabaseHandle {
    pub env: Env,
    pub database: Database<BEU32, Unspecified>,
    #[allow(unused)]
    pub tempdir: TempDir,
}

impl fmt::Display for DatabaseHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rtxn = self.env.read_txn().unwrap();
        for result in
            self.database.remap_data_type::<LazyDecode<NodeCodec<Angular>>>().iter(&rtxn).unwrap()
        {
            let (i, lazy_node) = result.unwrap();
            if i != u32::MAX {
                let node = lazy_node.decode().unwrap();
                writeln!(f, "{i}: {node:?}")?;
            } else {
                let roots_bytes =
                    self.database.remap_data_type::<ByteSlice>().get(&rtxn, &i).unwrap().unwrap();
                let roots: Vec<u32> = pod_collect_to_vec(roots_bytes);
                writeln!(f, "\nu32::MAX: {roots:?}")?;
            }
        }
        Ok(())
    }
}

fn create_database() -> DatabaseHandle {
    let dir = tempfile::tempdir().unwrap();
    let env = EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()).unwrap();
    let mut wtxn = env.write_txn().unwrap();
    let database: Database<BEU32, Unspecified> = env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();
    DatabaseHandle { env, database, tempdir: dir }
}

fn rng() -> StdRng {
    StdRng::from_seed(std::array::from_fn(|_| 42))
}
