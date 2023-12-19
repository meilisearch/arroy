use std::fmt;

use heed::types::LazyDecode;
use heed::{Env, EnvOpenOptions};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tempfile::TempDir;

use crate::distances::Angular;
use crate::{Database, Distance, MetadataCodec, NodeCodec, NodeMode};

mod angular;
mod reader;
mod writer;

pub struct DatabaseHandle<D> {
    pub env: Env,
    pub database: Database<D>,
    #[allow(unused)]
    pub tempdir: TempDir,
}

impl<D: Distance> fmt::Display for DatabaseHandle<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rtxn = self.env.read_txn().unwrap();

        let mut old_index;
        let mut current_index = None;
        let mut last_mode = NodeMode::Item;

        for result in
            self.database.remap_data_type::<LazyDecode<NodeCodec<Angular>>>().iter(&rtxn).unwrap()
        {
            let (key, lazy_node) = result.unwrap();

            old_index = current_index;
            current_index = Some(key.index);

            if old_index != current_index {
                writeln!(f, "==================")?;
                writeln!(f, "Dumping index {}", current_index.unwrap())?;
            }

            if last_mode != key.node.mode && key.node.mode == NodeMode::Item {
                writeln!(f)?;
                last_mode = key.node.mode;
            }

            match key.node.mode {
                NodeMode::Item => {
                    let node = lazy_node.decode().unwrap();
                    writeln!(f, "Item {}: {node:?}", key.node.item)?;
                }
                NodeMode::Tree => {
                    let node = lazy_node.decode().unwrap();
                    writeln!(f, "Tree {}: {node:?}", key.node.item)?;
                }
                NodeMode::Metadata => {
                    let metadata = self
                        .database
                        .remap_data_type::<MetadataCodec>()
                        .get(&rtxn, &key)
                        .unwrap()
                        .unwrap();
                    writeln!(f, "Root: {metadata:?}")?;
                }
            }
        }
        Ok(())
    }
}

fn create_database<D: Distance>() -> DatabaseHandle<D> {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();

    let dir = tempfile::tempdir().unwrap();
    let env = EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()).unwrap();
    let mut wtxn = env.write_txn().unwrap();
    let database: Database<D> = env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();
    DatabaseHandle { env, database, tempdir: dir }
}

fn rng() -> StdRng {
    StdRng::from_seed(std::array::from_fn(|_| 42))
}
