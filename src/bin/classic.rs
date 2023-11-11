use std::time::Instant;

use arroy::{ArroyReader, DistanceType};

fn main() -> std::io::Result<()> {
    let dimensions = 40;
    let distance_type = DistanceType::Angular;
    let tree = std::fs::read("test.tree").unwrap();

    let arroy = ArroyReader::new(&tree[..], dimensions, distance_type);
    // dbg!(&arroy);
    let v = arroy.item_vector(0).unwrap();
    println!("{v:?}");

    let before = Instant::now();
    let results = arroy.nns_by_item(0, 30, None).unwrap();
    eprintln!("It took {:.02?} to find the nns", before.elapsed());
    println!("{results:?}");

    Ok(())
}
