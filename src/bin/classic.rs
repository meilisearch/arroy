use arroy::{ArroyReader, DistanceType};

fn main() -> std::io::Result<()> {
    let dimensions = 40;
    let distance_type = DistanceType::Angular;
    let tree = std::fs::read("test.tree").unwrap();

    let arroy = ArroyReader::new(&tree[..], dimensions, distance_type);
    // dbg!(&arroy);
    let v = arroy.item_vector(0).unwrap();
    let results = arroy.nns_by_item(0, 3, None).unwrap();

    println!("{v:?}");
    println!("{results:?}");

    Ok(())
}
