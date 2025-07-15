use crate::distance::Distance;
use crate::distances::Euclidean;
use crate::writer::fit_in_memory;
use rand::rngs::StdRng;
use rand::SeedableRng;
use roaring::RoaringBitmap;

#[test]
fn test_empty_bitmap() {
    let mut rng = StdRng::seed_from_u64(28);
    let mut bitmap = RoaringBitmap::new();
    let result = fit_in_memory::<Euclidean, _>(1024 * 1024, &mut bitmap, 128, &mut rng);
    assert!(result.is_none());
}

#[test]
fn test_all_items_fit() {
    let mut rng = StdRng::seed_from_u64(35);
    let mut bitmap = RoaringBitmap::from_sorted_iter(0..100).unwrap();
    let result = fit_in_memory::<Euclidean, _>(usize::MAX, &mut bitmap, 128, &mut rng).unwrap();
    assert_eq!(result.len(), 100);
    assert!(bitmap.is_empty());
}

#[test]
fn test_less_items_than_dimensions() {
    let mut rng = StdRng::seed_from_u64(26);
    let mut bitmap = RoaringBitmap::from_sorted_iter(0..10).unwrap();
    let result = fit_in_memory::<Euclidean, _>(0, &mut bitmap, 128, &mut rng).unwrap();
    assert_eq!(result.len(), 10);
    assert!(bitmap.is_empty());
}

#[test]
fn test_partial_fit() {
    let mut rng = StdRng::seed_from_u64(3141592);
    let mut bitmap = RoaringBitmap::from_sorted_iter(0..1000).unwrap();

    let dimensions = 128;
    let largest_item_size = Euclidean::size_of_item(dimensions);
    let memory = largest_item_size * 500;

    let result = fit_in_memory::<Euclidean, _>(memory, &mut bitmap, dimensions, &mut rng).unwrap();
    // We can't assert properly on the len of the result because the page_size vary depending on the system
    assert!(result.len() > dimensions as u64);
    assert_eq!(1000, bitmap.len() + result.len());
}

#[test]
fn test_random_selection() {
    let mut rng = StdRng::seed_from_u64(24);
    let bitmap = RoaringBitmap::from_sorted_iter(0..1000).unwrap();

    let dimensions = 128;
    let largest_item_size = Euclidean::size_of_item(dimensions);
    let memory = largest_item_size * 500;

    // Get first batch
    let mut bitmap_clone = bitmap.clone();
    let result1 =
        fit_in_memory::<Euclidean, _>(memory, &mut bitmap_clone, dimensions, &mut rng).unwrap();
    assert!(result1.len() > dimensions as u64);
    assert_eq!(1000, bitmap_clone.len() + result1.len());

    // Get second batch
    let mut bitmap_clone = bitmap.clone();
    let result2 =
        fit_in_memory::<Euclidean, _>(memory, &mut bitmap_clone, dimensions, &mut rng).unwrap();
    assert!(result2.len() > dimensions as u64);
    assert_eq!(1000, bitmap_clone.len() + result2.len());

    // Batch must be different because of random selection but they must contains the same number of items
    assert_eq!(result1.len(), result2.len());
    assert_ne!(result1, result2);
}
