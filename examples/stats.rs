use std::fs;
use std::path::PathBuf;

use arroy::distances::DotProduct;
use arroy::{Database, Reader, Stats, TreeStats};
use clap::Parser;
use heed::EnvOpenOptions;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Sets a custom database path.
    #[arg(default_value = "import.ary")]
    database: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let Cli { database } = Cli::parse();

    let _ = fs::create_dir_all(&database);
    let env = EnvOpenOptions::new().open(&database)?;

    let rtxn = env.read_txn()?;
    let database: Database<DotProduct> = env.open_database(&rtxn, None)?.unwrap();
    let reader = Reader::open(&rtxn, 0, database)?;

    let mut dummy_sum = 0;
    let mut depth_sum = 0;
    let Stats { tree_stats } = reader.stats(&rtxn)?;
    let nb_roots = tree_stats.len();
    println!("There are {} trees in this arroy index.", nb_roots);
    for (i, TreeStats { depth, dummy_normals }) in tree_stats.into_iter().enumerate() {
        depth_sum += depth;
        dummy_sum += dummy_normals;
        println!("Tree {i} as a depth of {depth} and {dummy_normals} dummy normals");
    }

    println!(
        "On average, got a depth of {} and {} normals",
        depth_sum as f64 / nb_roots as f64,
        dummy_sum as f64 / nb_roots as f64
    );

    Ok(())
}