use std::borrow::Cow;
use std::time::Instant;

use arroy::{
    Distance, Euclidean, ItemId, Leaf, Reader, Result, Writer, ALIGNED_VECTOR, BEU32, NON_ALIGNED_VECTOR,
};
use heed::{Database, EnvOpenOptions, RwTxn, Unspecified};
use instant_distance::{Builder, HnswMap, MapItem};
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
    let (points, items_ids) = generate_points(rng_points, NUMBER_VECTORS, VECTOR_DIMENSIONS);
    eprintln!("took {:.02?} to generate the {NUMBER_VECTORS} random points", before.elapsed());

    let mut wtxn = env.write_txn()?;
    let database: Database<BEU32, Unspecified> = env.create_database(&mut wtxn, None)?;
    let before = Instant::now();
    load_into_arroy(rng_arroy, wtxn, database, VECTOR_DIMENSIONS, &points)?;
    eprintln!("took {:.02?} to load into arroy", before.elapsed());

    let before = Instant::now();
    let hnsw = load_into_hnsw(points, items_ids);
    eprintln!("took {:.02?} to load into hnsw", before.elapsed());

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

    let aligned = ALIGNED_VECTOR.load(std::sync::atomic::Ordering::SeqCst);
    let non_aligned = NON_ALIGNED_VECTOR.load(std::sync::atomic::Ordering::SeqCst);
    eprintln!("ALIGNED_VECTOR: {}, NON_ALIGNED_VECTOR: {}", aligned, non_aligned);
    eprintln!(
        "which means {:.02?}% of aligned reads",
        aligned as f64 / (aligned + non_aligned) as f64 * 100.0
    );

    let first = Point(reader.item_vector(&rtxn, 0)?.unwrap());

    println!();
    let before = Instant::now();
    let mut search = instant_distance::Search::default();
    for MapItem { distance, value, .. } in hnsw.search(&first, &mut search).take(NUMBER_FETCHED) {
        println!("id({}): distance({distance})", value);
    }
    eprintln!("took {:.02?} to find into hnsw", before.elapsed());

    // HeedReader::load_from_tree(&mut wtxn, database, dimensions, distance_type, &tree)?;

    Ok(())
}

fn load_into_arroy(
    rng: StdRng,
    mut wtxn: RwTxn,
    database: Database<BEU32, Unspecified>,
    dimensions: usize,
    points: &[Point],
) -> Result<()> {
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, database, dimensions)?;
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
        let p = Leaf { header: Euclidean::new_header(&self.0), vector: Cow::Borrowed(&self.0) };
        let q = Leaf { header: Euclidean::new_header(&other.0), vector: Cow::Borrowed(&other.0) };
        arroy::Euclidean::distance(&p, &q).sqrt()
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
