use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ordered_float::OrderedFloat;
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, Fill, Rng, SeedableRng};
use std::{cmp::Reverse, collections::BinaryHeap, hint::black_box};

fn min_heap_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("min heap");
    let mut rng = StdRng::seed_from_u64(42);

    let uniform = Uniform::new(0.0f32, 1.0);

    for k in vec![10, 100, 1000] {
        let mut data: Vec<Reverse<(OrderedFloat<f32>, u32)>> = (0..1000)
            .map(|i| {
                let dist = uniform.sample(&mut rng);
                Reverse((OrderedFloat(dist), i as u32))
            })
            .collect();

        group.bench_function(BenchmarkId::new("everything in", k), move |b| {
            b.iter_batched(
                || data.clone(),
                |data| {
                    // NOTE: we should NOT shove everything into a binary heap (bad)
                    let mut min_heap = BinaryHeap::from(data);
                    let mut output = Vec::with_capacity(k);

                    while let Some(Reverse((OrderedFloat(dist), item))) = min_heap.pop() {
                        if output.len() == k {
                            break;
                        }
                        output.push((item, 0.0f32));
                    }
                },
                criterion::BatchSize::LargeInput,
            );
        });

        // TODO: test with iterative take k approach
    }
}

criterion_group!(benches, min_heap_top_k);
criterion_main!(benches);
