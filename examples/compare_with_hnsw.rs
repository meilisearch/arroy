use std::time::Instant;

use arroy::{Euclidean, Reader, Writer, BEU32};
use heed::{Database, EnvOpenOptions, RwTxn, Unspecified};
use instant_distance::{Builder, Hnsw, Item, PointId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024;
const NUMBER_VECTORS: usize = 10_000;
const NUMBER_FETCHED: usize = 10;

fn main() -> heed::Result<()> {
    let dir = tempfile::tempdir()?;
    let env = EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;

    let dimensions = 300;
    let rng_points = StdRng::seed_from_u64(42);
    let rng_arroy = rng_points.clone();

    let points = generate_points(rng_points, NUMBER_VECTORS, dimensions);

    let mut wtxn = env.write_txn()?;
    let database: Database<BEU32, Unspecified> = env.create_database(&mut wtxn, None)?;
    let before = Instant::now();
    load_into_arroy(rng_arroy, wtxn, database, dimensions, &points)?;
    eprintln!("took {:.02?} to load into arroy", before.elapsed());

    let before = Instant::now();
    let hnsw = load_into_hnsw(points);
    eprintln!("took {:.02?} to load into hnsw", before.elapsed());

    let before = Instant::now();
    let rtxn = env.read_txn()?;
    let reader = Reader::<Euclidean>::open(&rtxn, database, dimensions)?;
    for (id, dist) in reader.nns_by_item(&rtxn, 0, NUMBER_FETCHED, None)?.unwrap() {
        println!("id({id}): distance({dist})");
    }
    eprintln!("took {:.02?} to find into arroy", before.elapsed());

    let point = Point(reader.item_vector(&rtxn, 0)?.unwrap());

    println!();
    let before = Instant::now();
    let mut search = instant_distance::Search::default();
    for Item { distance, pid, point: _ } in hnsw.search(&point, &mut search).take(NUMBER_FETCHED) {
        println!("id({}): distance({distance})", pid.into_inner());
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

fn load_into_hnsw(points: Vec<Point>) -> (Hnsw<Point>, Vec<PointId>) {
    Builder::default().seed(42).build_hnsw(points)
}

#[derive(Debug, Clone)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.0.iter().zip(other.0.iter()).map(|(&p, &q)| (p - q) * (p - q)).sum()
    }
}

fn generate_points<R: Rng>(mut rng: R, count: usize, dimensions: usize) -> Vec<Point> {
    let mut points = Vec::with_capacity(count);
    for _ in 0..count {
        let mut vector = vec![0.0; dimensions];
        rng.try_fill(&mut vector[..]).unwrap();
        points.push(Point(vector));
    }
    points
}
