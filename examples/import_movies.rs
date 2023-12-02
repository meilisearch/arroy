//! Download the associated file at https://www.notion.so/meilisearch/Movies-embeddings-1de3258859f54b799b7883882219d266

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use arroy::distances::DotProduct;
use arroy::{Database, Writer};
use clap::Parser;
use heed::EnvOpenOptions;
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

    /// The seed to generate the internal trees.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<(), heed::BoxedError> {
    let Cli { database, map_size, dimensions, seed } = Cli::parse();

    let mut rng = StdRng::seed_from_u64(seed);
    let reader = BufReader::new(std::io::stdin());

    let _ = fs::create_dir_all(&database);
    let env = EnvOpenOptions::new().map_size(map_size).open(&database).unwrap();
    let mut wtxn = env.write_txn().unwrap();
    let database: Database<DotProduct> = env.create_database(&mut wtxn, None)?;
    let writer = Writer::<DotProduct>::prepare(&mut wtxn, database, 0, dimensions)?;

    // The file look like that
    // === BEGIN vectors ===
    // 0, [0.010056925, -0.0045358953, 0.009904552, 0.0046241777, ..., -0.050245073]
    // === END vectors ===

    let now = Instant::now();
    let mut count = 0;
    for line in reader.lines() {
        let line = line?;
        if line.starts_with("===") {
            continue;
        }

        let (id, vector) = line.split_once(',').expect(&line);
        let id: u32 = id.parse()?;
        let vector: Vec<_> = vector
            .trim_matches(|c: char| c.is_whitespace() || c == '[' || c == ']')
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap())
            .collect();

        assert_eq!(vector.len(), dimensions);
        writer.add_item(&mut wtxn, id, &vector)?;
        count += 1;
    }
    println!("Took {:.2?} to parse and insert into arroy", now.elapsed());
    println!("There are {count} vectors");
    println!();

    println!("Building the arroy internal trees...");
    let now = Instant::now();
    for (id, vector) in vectors.iter() {
        writer.add_item(&mut wtxn, *id, vector).unwrap();
    }
    let insert = now.elapsed();

    wtxn.commit().unwrap();

    let mut wtxn = env.write_txn().unwrap();
    writer.build_in_parallel(&mut wtxn, rng, Some(111)).unwrap();
    wtxn.commit().unwrap();
    println!("Took {:.2?} to build", now.elapsed());

    Ok(())
}
