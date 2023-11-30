use std::fmt;

use heed::types::LazyDecode;
use heed::{Database, Env, EnvOpenOptions, Unspecified};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tempfile::TempDir;

use crate::{Angular, ItemId, KeyCodec, MetadataCodec, NodeCodec};

mod reader;
mod writer;

pub struct DatabaseHandle {
    pub env: Env,
    pub database: Database<KeyCodec, Unspecified>,
    #[allow(unused)]
    pub tempdir: TempDir,
}

impl fmt::Display for DatabaseHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rtxn = self.env.read_txn().unwrap();

        let mut old_index;
        let mut current_index = None;
        let mut last_mode = 0;

        for result in
            self.database.remap_data_type::<LazyDecode<NodeCodec<Angular>>>().iter(&rtxn).unwrap()
        {
            let (key, lazy_node) = result.unwrap();

            old_index = current_index;
            current_index = Some(key.index & 0b1111_1110);

            if old_index != current_index {
                writeln!(f, "==================")?;
                writeln!(f, "Dumping index {}", current_index.unwrap())?;
            }

            if last_mode != key.index && key.index & 1 == 0 {
                writeln!(f)?;
                last_mode = key.index;
            }

            match key.index & 1 {
                0 => {
                    let node = lazy_node.decode().unwrap();
                    writeln!(f, "Item {}: {node:?}", key.item)?;
                }
                1 if key.item == ItemId::MAX => {
                    let metadata = self
                        .database
                        .remap_data_type::<MetadataCodec>()
                        .get(&rtxn, &key)
                        .unwrap()
                        .unwrap();
                    writeln!(f, "Root: {metadata:?}")?;
                }
                1 => {
                    let node = lazy_node.decode().unwrap();
                    writeln!(f, "Tree {}: {node:?}", key.item)?;
                }
                _ => unreachable!("Impossiburuu"),
            }
        }
        Ok(())
    }
}

fn create_database() -> DatabaseHandle {
    let dir = tempfile::tempdir().unwrap();
    let env = EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()).unwrap();
    let mut wtxn = env.write_txn().unwrap();
    let database: Database<KeyCodec, Unspecified> = env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();
    DatabaseHandle { env, database, tempdir: dir }
}

fn rng() -> StdRng {
    StdRng::from_seed(std::array::from_fn(|_| 42))
}
