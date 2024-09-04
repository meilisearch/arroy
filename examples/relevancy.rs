use std::fmt;

use rand::seq::SliceRandom;

use arroy::distances::{
    Angular, BinaryQuantizedAngular, BinaryQuantizedEuclidean, BinaryQuantizedManhattan,
    DotProduct, Euclidean, Manhattan,
};
use arroy::internals::{self, Leaf, NodeCodec, UnalignedVector};
use arroy::{Database, Distance, ItemId, Result, Writer};
use heed::{EnvOpenOptions, RwTxn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;
const NUMBER_VECTORS: usize = 4_000;
const OVERSAMPLING: usize = 3;

fn main() {
    let dimensions_tested = [256, 512, 1024, 1536, 3072];
    let recall_tested = [1, 10, 50, 100];
    let color: Vec<_> = (0..=100).step_by(10).map(|i| Recall(i as f32 / 100.0)).collect();

    println!("Testing the following dimensions: {dimensions_tested:?}");
    println!("Testing the following recall: @{recall_tested:?}");
    println!("Oversampling of: x{OVERSAMPLING}");
    println!("With color code: {color:?}");
    println!("Starting...");
    println!();

    for (distance_name, func) in &[
        (
            BinaryQuantizedAngular::name(),
            &measure_distance::<BinaryQuantizedAngular, Angular> as &dyn Fn(usize, usize) -> f32,
        ),
        (Angular::name(), &measure_distance::<Angular, Angular> as &dyn Fn(usize, usize) -> f32),
        (
            BinaryQuantizedManhattan::name(),
            &measure_distance::<BinaryQuantizedManhattan, Manhattan>
                as &dyn Fn(usize, usize) -> f32,
        ),
        (
            Manhattan::name(),
            &measure_distance::<Manhattan, Manhattan> as &dyn Fn(usize, usize) -> f32,
        ),
        (
            BinaryQuantizedEuclidean::name(),
            &measure_distance::<BinaryQuantizedEuclidean, Euclidean>
                as &dyn Fn(usize, usize) -> f32,
        ),
        (
            Euclidean::name(),
            &measure_distance::<Euclidean, Euclidean> as &dyn Fn(usize, usize) -> f32,
        ),
        (
            DotProduct::name(),
            &measure_distance::<DotProduct, DotProduct> as &dyn Fn(usize, usize) -> f32,
        ),
    ] {
        let now = std::time::Instant::now();
        println!("{distance_name}");
        // The openAI dimensions
        for dimensions in [256, 512, 1024, 1536, 3072] {
            let mut recall = Vec::new();
            for number_fetched in recall_tested {
                let rec = (func)(number_fetched, dimensions);
                recall.push(Recall(rec));
            }
            println!("For {dimensions:4} dim, recall: {recall:3?}");
        }
        println!("Took {:?}", now.elapsed());
        println!();
    }
}

struct Recall(f32);

impl fmt::Debug for Recall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            // red
            f32::NEG_INFINITY..=0.25 => write!(f, "\x1b[1;31m")?,
            // yellow
            0.25..=0.5 => write!(f, "\x1b[1;33m")?,
            // green
            0.5..=0.75 => write!(f, "\x1b[1;32m")?,
            // blue
            0.75..=0.90 => write!(f, "\x1b[1;34m")?,
            // cyan
            0.90..=0.999 => write!(f, "\x1b[1;36m")?,
            // underlined cyan
            0.999..=f32::INFINITY => write!(f, "\x1b[1;4;36m")?,
            _ => (),
        }
        write!(f, "{:.2}\x1b[0m", self.0)
    }
}

fn measure_distance<ArroyDistance: Distance, PerfectDistance: Distance>(
    number_fetched: usize,
    dimensions: usize,
) -> f32 {
    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path()) }.unwrap();

    let mut rng = StdRng::seed_from_u64(13);
    let points = generate_points(&mut rng, NUMBER_VECTORS, dimensions);
    let mut wtxn = env.write_txn().unwrap();

    let database = env
        .create_database::<internals::KeyCodec, NodeCodec<ArroyDistance>>(&mut wtxn, None)
        .unwrap();
    load_into_arroy(&mut rng, &mut wtxn, database, dimensions, &points).unwrap();

    let reader = arroy::Reader::open(&wtxn, 0, database).unwrap();

    let mut correctly_retrieved = 0;
    for _ in 0..100 {
        let querying = points.choose(&mut rng).unwrap();

        let relevant = partial_sort_by::<PerfectDistance>(
            points.iter().map(|(i, v)| (*i, v.as_slice())),
            &querying.1,
            number_fetched,
        );

        let mut arroy = reader
            .nns_by_item(&wtxn, querying.0, number_fetched * OVERSAMPLING, None, None, None)
            .unwrap()
            .unwrap();
        arroy.truncate(number_fetched);

        for ret in arroy {
            if relevant.iter().any(|(id, _, _)| *id == ret.0) {
                correctly_retrieved += 1;
            }
        }
    }

    // println!("recall@{number_fetched}: {}", correctly_retrieved as f32 / relevant.len() as f32);
    correctly_retrieved as f32 / (number_fetched as f32 * 100.0)
}

fn partial_sort_by<'a, D: Distance>(
    mut vectors: impl Iterator<Item = (ItemId, &'a [f32])>,
    sort_by: &[f32],
    elements: usize,
) -> Vec<(ItemId, &'a [f32], f32)> {
    let mut ret = Vec::with_capacity(elements);
    ret.extend(vectors.by_ref().take(elements).map(|(i, v)| (i, v, distance::<D>(sort_by, v))));
    ret.sort_by(|(_, _, left), (_, _, right)| left.total_cmp(right));

    if ret.is_empty() {
        return ret;
    }

    for (item_id, vector) in vectors {
        let distance = distance::<D>(sort_by, vector);
        if distance < ret.last().unwrap().2 {
            match ret.binary_search_by(|(_, _, d)| d.total_cmp(&distance)) {
                Ok(i) | Err(i) => {
                    ret.pop();
                    ret.insert(i, (item_id, vector, distance))
                }
            }
        }
    }

    ret
}

fn distance<D: Distance>(left: &[f32], right: &[f32]) -> f32 {
    let left = UnalignedVector::from_slice(left);
    let left = Leaf { header: D::new_header(&left), vector: left };
    let right = UnalignedVector::from_slice(right);
    let right = Leaf { header: D::new_header(&right), vector: right };

    D::built_distance(&left, &right)
}

fn load_into_arroy<D: Distance>(
    rng: &mut StdRng,
    wtxn: &mut RwTxn,
    database: Database<D>,
    dimensions: usize,
    points: &[(ItemId, Vec<f32>)],
) -> Result<()> {
    let writer = Writer::<D>::new(database, 0, dimensions);
    for (i, vector) in points.iter() {
        writer.add_item(wtxn, *i, &vector[..])?;
    }
    writer.build(wtxn, rng, None)?;

    Ok(())
}

fn generate_points<R: Rng>(mut rng: R, count: usize, dimensions: usize) -> Vec<(ItemId, Vec<f32>)> {
    let mut points = Vec::with_capacity(count);
    for item_id in 0..count {
        let mut vector = vec![0.0; dimensions];
        for scalar in &mut vector {
            *scalar = rng.gen_range(-1.0..1.0);
        }
        // rng.try_fill(&mut vector[..]).unwrap();
        points.push((item_id.try_into().unwrap(), vector));
    }
    points
}
