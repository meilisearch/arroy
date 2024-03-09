//! Download the associated file at https://www.notion.so/meilisearch/Movies-embeddings-1de3258859f54b799b7883882219d266

use std::fs;
use std::io::{BufReader, ErrorKind, Read};
use std::path::PathBuf;
use std::time::Instant;

use arroy::distances::DotProduct;
use arroy::{Database, Writer};
use clap::Parser;
use heed::{EnvFlags, EnvOpenOptions};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// 2 GiB
const DEFAULT_MAP_SIZE: usize = 1024 * 1024 * 1024 * 2;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Sets a custom database path.
    #[arg(default_value = "import.ary")]
    database: PathBuf,

    /// Specify the size of the database.
    #[arg(long, default_value_t = DEFAULT_MAP_SIZE)]
    map_size: usize,

    /// The number of dimensions to construct the arroy tree.
    #[arg(long, default_value_t = 768)]
    dimensions: usize,

    /// Use the MDB_WRITEMAP option to reduce the memory usage of LMDB.
    #[arg(long)]
    write_map: bool,

    /// Do not try to append items into the database.
    #[arg(long)]
    no_append: bool,

    /// The number of tress to generate.
    #[arg(long)]
    n_trees: Option<usize>,

    /// The seed to generate the internal trees.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<(), heed::BoxedError> {
    env_logger::init();

    let Cli { database, map_size, dimensions, write_map, no_append, n_trees, seed } = Cli::parse();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut reader = BufReader::new(std::io::stdin());

    let _ = fs::create_dir_all(&database);

    // Open the environment with the appropriate flags.
    let flags = if write_map { EnvFlags::WRITE_MAP } else { EnvFlags::empty() };
    let mut env_builder = EnvOpenOptions::new();
    env_builder.map_size(map_size);
    unsafe { env_builder.flags(flags) };
    let env = env_builder.open(&database).unwrap();

    let mut wtxn = env.write_txn().unwrap();
    let database: Database<DotProduct> = env.create_database(&mut wtxn, None)?;
    let writer = Writer::<DotProduct>::new(database, 0, dimensions);

    // The file look like that
    // === BEGIN vectors ===
    // 0, [0.010056925, -0.0045358953, 0.009904552, 0.0046241777, ..., -0.050245073]
    // === END vectors ===

    let now = Instant::now();
    let mut vector = Vec::with_capacity(dimensions);
    let mut count = 0;
    loop {
        let mut block = [0; 4];
        let id = match reader.read_exact(&mut block) {
            Ok(()) => u32::from_be_bytes(block),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };

        vector.clear();
        for _ in 0..dimensions {
            reader.read_exact(&mut block)?;
            vector.push(f32::from_be_bytes(block));
        }

        if no_append {
            writer.add_item(&mut wtxn, id, &vector)?;
        } else {
            writer.append_item(&mut wtxn, id, &vector)?;
        }

        count += 1;
    }
    println!("Took {:.2?} to parse and insert into arroy", now.elapsed());
    println!("There are {count} vectors");
    println!();

    println!("Building the arroy internal trees...");
    let now = Instant::now();
    writer.build(&mut wtxn, &mut rng, n_trees).unwrap();
    wtxn.commit().unwrap();
    println!("Took {:.2?} to build", now.elapsed());

    Ok(())
}
