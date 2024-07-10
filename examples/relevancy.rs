use rand::seq::SliceRandom;

use arroy::distances::{Angular, BinaryQuantizedEuclidean, DotProduct, Euclidean, Manhattan};
use arroy::internals::{self, Leaf, NodeCodec, UnalignedVector};
use arroy::{Database, Distance, ItemId, Result, Writer};
use heed::{EnvOpenOptions, RwTxn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;

const NUMBER_VECTORS: usize = 4_000;
// The openAI dimensions
const VECTOR_DIMENSIONS: usize = 256;
// const VECTOR_DIMENSIONS: usize = 512;
// const VECTOR_DIMENSIONS: usize = 1024;
// const VECTOR_DIMENSIONS: usize = 1536;
// const VECTOR_DIMENSIONS: usize = 3072;

fn main() {
    for (distance_name, func) in &[
        (Angular::name(), &measure_distance::<Angular, Angular> as &dyn Fn(usize)),
        (Euclidean::name(), &measure_distance::<Euclidean, Euclidean> as &dyn Fn(usize)),
        (Manhattan::name(), &measure_distance::<Manhattan, Manhattan> as &dyn Fn(usize)),
        (DotProduct::name(), &measure_distance::<DotProduct, DotProduct> as &dyn Fn(usize)),
        (
            BinaryQuantizedEuclidean::name(),
            &measure_distance::<BinaryQuantizedEuclidean, Euclidean> as &dyn Fn(usize),
        ),
    ] {
        println!("{distance_name}");
        for number_fetched in [1, 10, 50, 100] {
            (func)(number_fetched);
        }
        println!();
    }
}

fn measure_distance<ArroyDistance: Distance, PerfectDistance: Distance>(number_fetched: usize) {
    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path()) }.unwrap();

    let mut rng = StdRng::seed_from_u64(13);
    let points = generate_points(&mut rng, NUMBER_VECTORS, VECTOR_DIMENSIONS);
    let mut wtxn = env.write_txn().unwrap();

    let database = env
        .create_database::<internals::KeyCodec, NodeCodec<ArroyDistance>>(&mut wtxn, None)
        .unwrap();
    load_into_arroy(&mut rng, &mut wtxn, database, VECTOR_DIMENSIONS, &points).unwrap();

    let reader = arroy::Reader::open(&wtxn, 0, database).unwrap();

    let querying = points.choose(&mut rng).unwrap();

    let relevant = partial_sort_by::<PerfectDistance>(
        points.iter().map(|(i, v)| (*i, v.as_slice())),
        &querying.1,
        number_fetched,
    );

    let arroy = reader.nns_by_item(&wtxn, querying.0, number_fetched, None, None).unwrap().unwrap();

    let mut correctly_retrieved = 0;
    for ret in arroy {
        if relevant.iter().any(|(id, _, _)| *id == ret.0) {
            correctly_retrieved += 1;
        }
    }

    println!("recall@{number_fetched}: {}", correctly_retrieved as f32 / relevant.len() as f32);
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
