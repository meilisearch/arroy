use std::num::NonZeroUsize;
use std::time::Instant;

use arroy::{Euclidean, ItemId, Reader, Writer, BEU32};
use heed::{Database, EnvOpenOptions, RwTxn, Unspecified};
use instant_distance::{Builder, HnswMap, MapItem};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024;
const NUMBER_VECTORS: usize = 10_000;
const VECTOR_DIMENSIONS: usize = 300;
const NUMBER_FETCHED: usize = 10;

fn main() -> heed::Result<()> {
    let dir = tempfile::tempdir()?;
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
    let reader = Reader::<Euclidean>::open(&rtxn, database, VECTOR_DIMENSIONS)?;

    // By macking it precise we are near the HNSW but
    // we take a lot more time to search than the HNSW.
    let is_precise = true;
    let search_k =
        if is_precise { NonZeroUsize::new(NUMBER_FETCHED * reader.n_trees() * 20) } else { None };

    for (id, dist) in reader.nns_by_item(&rtxn, 0, NUMBER_FETCHED, search_k)?.unwrap() {
        println!("id({id}): distance({dist})");
    }
    eprintln!("took {:.02?} to find into arroy", before.elapsed());

    let point = Point(reader.item_vector(&rtxn, 0)?.unwrap());

    println!();
    let before = Instant::now();
    let mut search = instant_distance::Search::default();
    for MapItem { distance, value, .. } in hnsw.search(&point, &mut search).take(NUMBER_FETCHED) {
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
) -> heed::Result<()> {
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, database, dimensions)?;
    for (i, Point(vector)) in points.iter().enumerate() {
        writer.add_item(&mut wtxn, i.try_into().unwrap(), &vector[..])?;
    }
    writer.build(&mut wtxn, rng, None)?;
    wtxn.commit()
}

fn load_into_hnsw(points: Vec<Point>, items_ids: Vec<ItemId>) -> HnswMap<Point, ItemId> {
    Builder::default().seed(42).build(points, items_ids)
}

#[derive(Debug, Clone)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.0.iter().zip(other.0.iter()).map(|(&p, &q)| (p - q) * (p - q)).sum::<f32>().sqrt()
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
