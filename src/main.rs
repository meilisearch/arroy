use annoy_rs::*;
use arroy::{ArroyReader, DistanceType};

fn main() {
    let dimensions = 40;
    let distance_type = DistanceType::Angular;
    let tree = std::fs::read("test.tree").unwrap();
    let arroy = ArroyReader::new(&tree[..], dimensions, distance_type);

    // dbg!(&arroy);
    let v = arroy.item_vector(0).unwrap();
    let results = arroy.nns_by_item(0, 3, None).unwrap();
    // println!("{v:?}");

    let index = AnnoyIndex::load(40, "test.tree", IndexType::Angular).unwrap();
    // dbg!(&index);
    let v0 = index.get_item_vector(0);
    let results0 = index.get_nearest_to_item(0, 3, -1, true);
    // println!("{v0:?}");

    assert_eq!(v, v0);

    assert_eq!(results[0].0, results0.id_list[0] as usize);
    assert_eq!(results[1].0, results0.id_list[1] as usize);
    assert_eq!(results[2].0, results0.id_list[2] as usize);

    assert_eq!(results[0].1, results0.distance_list[0]);
    assert_eq!(results[1].1, results0.distance_list[1]);
    assert_eq!(results[2].1, results0.distance_list[2]);
}
