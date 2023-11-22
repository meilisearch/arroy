use std::time::Instant;

use arroy::{Euclidean, Reader, Result, Writer, BEU32};
use heed::{Database, EnvOpenOptions, Unspecified};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;
const NUMBER_VECTORS: usize = 60_000;
const VECTOR_DIMENSIONS: usize = 768;
const NUMBER_FETCHED: usize = 5;

fn main() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

    let rng_points = StdRng::seed_from_u64(42);
    let rng_arroy = rng_points.clone();

    let before = Instant::now();
    let points = generate_points(rng_points, NUMBER_VECTORS, VECTOR_DIMENSIONS);
    eprintln!("took {:.02?} to generate the {NUMBER_VECTORS} random points", before.elapsed());

    // Reset the measurements
    let _ = arroy::get_and_reset_measurements();

    let mut wtxn = env.write_txn()?;
    let database: Database<BEU32, Unspecified> = env.create_database(&mut wtxn, None)?;
    let before = Instant::now();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, database, VECTOR_DIMENSIONS)?;
    for (i, vector) in points.iter().enumerate() {
        writer.add_item(&mut wtxn, i.try_into().unwrap(), &vector[..])?;
    }
    writer.build(&mut wtxn, rng_arroy, None)?;
    wtxn.commit()?;
    eprintln!("took {:.02?} to load into arroy", before.elapsed());

    let m = arroy::get_and_reset_measurements();
    eprintln!(
        "(insertion) which means {:.02?}% of aligned reads",
        m.aligned_slices_read as f64 / (m.aligned_slices_read + m.unaligned_slices_read) as f64
            * 100.0
    );

    let before = Instant::now();
    let rtxn = env.read_txn()?;
    let reader = Reader::<Euclidean>::open(&rtxn, database)?;

    // By making it precise we are near the HNSW but
    // we take a lot more time to search than the HNSW.
    let is_precise = false;
    let search_k = if is_precise { reader.n_nodes(&rtxn).unwrap() } else { None };

    for (id, dist) in reader.nns_by_item(&rtxn, 0, NUMBER_FETCHED, search_k)?.unwrap() {
        println!("id({id}): distance({dist})");
    }
    eprintln!("took {:.02?} to find into arroy", before.elapsed());

    let m = arroy::get_and_reset_measurements();
    eprintln!(
        "(search) which means {:.02?}% of aligned reads",
        m.aligned_slices_read as f64 / (m.aligned_slices_read + m.unaligned_slices_read) as f64
            * 100.0
    );

    Ok(())
}

fn generate_points<R: Rng>(mut rng: R, count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    let mut points = Vec::with_capacity(count);
    for _item_id in 0..count {
        let mut vector = vec![0.0; dimensions];
        rng.try_fill(&mut vector[..]).unwrap();
        points.push(vector);
    }
    points
}
