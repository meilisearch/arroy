use std::fs::File;
use std::io::BufWriter;

use arroy::distances::DotProduct;
use arroy::{Database, Reader};
use heed::EnvOpenOptions;

fn main() {
    let mut args = std::env::args();
    let dir_path = args.nth(1).expect("Provide the path to a database");
    let output_file = args.nth(1).unwrap_or_else(|| {
        eprintln!("No output path is specified. Writing to `graph.dot` by default.");
        "graph.dot".to_string()
    });

    let output = File::create(&output_file).unwrap();
    let writer = BufWriter::new(output);

    let env = EnvOpenOptions::new()
        .map_size(1024 * 1024 * 1024 * 2) // 2GiB
        .open(dir_path)
        .unwrap();

    let rtxn = env.read_txn().unwrap();
    let database: Database<DotProduct> =
        env.database_options().types().open(&rtxn).unwrap().unwrap();

    let reader = Reader::<DotProduct>::open(&rtxn, 0, database).unwrap();

    reader.plot_internals(&rtxn, writer).unwrap();

    eprintln!("To convert the graph to a png, run: `dot {} -T png > graph.png`", output_file);
    eprintln!("`dot` comes from the graphiz package");
}
