use std::iter;

use approx::assert_relative_eq;
use rand::Rng;

use super::*;
use crate::internals::UnalignedF32Slice;
use crate::spaces::simple::dot_product_non_optimized;
use crate::{ItemId, Reader, Writer};

#[allow(clippy::needless_lifetimes)]
fn generate_random_vectors<'r, R: Rng>(
    rng: &'r mut R,
    dimensions: usize,
) -> impl Iterator<Item = Vec<f32>> + 'r {
    iter::from_fn(move || {
        let mut vector = vec![0.0; dimensions];
        rng.try_fill(&mut vector[..]).unwrap();
        Some(vector)
    })
}

fn combinations(length: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..length.saturating_sub(1))
        .flat_map(move |i| ((i + 1)..length.saturating_sub(1)).map(move |j| (i, j)))
}

fn compute_angular_distance(p: &[f32], q: &[f32]) -> f32 {
    let p = UnalignedF32Slice::from_slice(p);
    let q = UnalignedF32Slice::from_slice(q);

    let pdot = dot_product_non_optimized(p, p);
    let qdot = dot_product_non_optimized(q, q);
    let pqdot = dot_product_non_optimized(p, q);
    let pn = pdot.sqrt();
    let qn = qdot.sqrt();
    let pnqn = pn * qn;
    if pnqn != 0.0 {
        let cos = pqdot / pnqn;
        (1.0 - cos) / 2.0
    } else {
        0.0
    }
}

#[test]
fn compare_local_and_arroy_distances() {
    let dimensions = 126;
    let mut rng = rng();
    let vectors: Vec<_> = generate_random_vectors(&mut rng, dimensions).take(100).collect();

    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::prepare(&mut wtxn, handle.database, 0, dimensions).unwrap();

    for (i, vec) in vectors.iter().enumerate() {
        writer.add_item(&mut wtxn, i as ItemId, vec).unwrap();
    }
    writer.build(&mut wtxn, &mut rng, Some(20)).unwrap();

    let reader = Reader::<Angular>::open(&wtxn, 0, handle.database).unwrap();
    for (i, j) in combinations(vectors.len()) {
        let arroy_distance =
            reader.distance_by_items(&wtxn, i as ItemId, j as ItemId).unwrap().unwrap();
        let local_distance = compute_angular_distance(&vectors[i], &vectors[j]);
        assert_relative_eq!(arroy_distance, local_distance, epsilon = f32::EPSILON * 1.8);
    }
}
