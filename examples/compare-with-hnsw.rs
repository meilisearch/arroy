use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::time::Instant;

use arroy::distances::Euclidean;
use arroy::internals::{Leaf, UnalignedF32Slice};
use arroy::{Database, Distance, ItemId, Reader, Result, Writer};
use heed::{EnvOpenOptions, RwTxn};
use instant_distance::{Builder, HnswMap, MapItem};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;
const NUMBER_VECTORS: usize = 4000;
const VECTOR_DIMENSIONS: usize = 768;
const NUMBER_FETCHED: usize = 5;

fn main() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

    let rng_points = StdRng::seed_from_u64(42);
    let mut rng_arroy = rng_points.clone();

    let before = Instant::now();
    let (points, items_ids) = generate_points(rng_points, NUMBER_VECTORS, VECTOR_DIMENSIONS);
    eprintln!("took {:.02?} to generate the {NUMBER_VECTORS} random points", before.elapsed());

    let mut wtxn = env.write_txn()?;
    let database: Database<Euclidean> = env.create_database(&mut wtxn, None)?;
    let before = Instant::now();
    load_into_arroy(&mut rng_arroy, wtxn, database, VECTOR_DIMENSIONS, &points)?;
    eprintln!("took {:.02?} to load into arroy", before.elapsed());

    let before = Instant::now();
    let hnsw = load_into_hnsw(points, items_ids);
    eprintln!("took {:.02?} to load into hnsw", before.elapsed());

    let before = Instant::now();
    let rtxn = env.read_txn()?;
    let reader = Reader::<Euclidean>::open(&rtxn, 0, database)?;

    // By making it precise we are near the HNSW but
    // we take a lot more time to search than the HNSW.
    let is_precise = true;
    let search_k =
        if is_precise { NonZeroUsize::new(NUMBER_FETCHED * reader.n_trees() * 20) } else { None };

    let arroy_results = reader.nns_by_item(&rtxn, 0, NUMBER_FETCHED, search_k, None)?.unwrap();
    eprintln!("took {:.02?} to find into arroy", before.elapsed());

    let first = Point(reader.item_vector(&rtxn, 0)?.unwrap());

    let before = Instant::now();
    let mut search = instant_distance::Search::default();
    let hnsw_results: Vec<_> = hnsw
        .search(&first, &mut search)
        .take(NUMBER_FETCHED)
        .map(|MapItem { distance, value, .. }| (value, distance))
        .collect();
    eprintln!("took {:.02?} to find into hnsw", before.elapsed());

    println!("arroy\t\t\t\t\tHNSW");
    for (arroy, hnsw) in arroy_results.into_iter().zip(hnsw_results) {
        println!("id({}) distance({})\tid({}) distance({})", arroy.0, arroy.1, hnsw.0, hnsw.1);
    }

    Ok(())
}

fn load_into_arroy(
    rng: &mut StdRng,
    mut wtxn: RwTxn,
    database: Database<Euclidean>,
    dimensions: usize,
    points: &[Point],
) -> Result<()> {
    let writer = Writer::<Euclidean>::new(database, 0, dimensions);
    for (i, Point(vector)) in points.iter().enumerate() {
        writer.add_item(&mut wtxn, i.try_into().unwrap(), &vector[..])?;
    }
    writer.build(&mut wtxn, rng, None)?;
    wtxn.commit()?;

    Ok(())
}

fn load_into_hnsw(points: Vec<Point>, items_ids: Vec<ItemId>) -> HnswMap<Point, ItemId> {
    Builder::default().seed(42).build(points, items_ids)
}

#[derive(Debug, Clone)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        let this = UnalignedF32Slice::from_slice(&self.0);
        let other = UnalignedF32Slice::from_slice(&other.0);
        let p = Leaf { header: Euclidean::new_header(this), vector: Cow::Borrowed(this) };
        let q = Leaf { header: Euclidean::new_header(other), vector: Cow::Borrowed(other) };
        arroy::distances::Euclidean::built_distance(&p, &q).sqrt()
    }
}

fn generate_points<R: Rng>(
    mut rng: R,
    count: usize,
    dimensions: usize,
) -> (Vec<Point>, Vec<ItemId>) {
    let mut points = Vec::with_capacity(count);
    let mut item_ids = Vec::with_capacity(count);
    for item_id in 0..count {
        let mut vector = vec![0.0; dimensions];
        rng.try_fill(&mut vector[..]).unwrap();
        points.push(Point(vector));
        item_ids.push(item_id.try_into().unwrap());
    }
    (points, item_ids)
}
