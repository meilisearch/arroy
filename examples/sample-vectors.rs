use std::io::{self, BufRead, BufReader, BufWriter, Write};

use clap::Parser;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Specify the number of items to generate.
    #[arg(default_value_t = 1_000)]
    count: usize,

    #[arg(long, default_value_t = 0)]
    first_id: u32,

    /// The seed to generate the internal trees.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> anyhow::Result<()> {
    let Cli { count, first_id, seed } = Cli::parse();
    let reader = BufReader::new(std::io::stdin());
    let mut vectors: Vec<(u32, Vec<f32>)> = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with("===") {
            continue;
        }

        let (id, vector) = line.split_once(',').expect(&line);
        let id: u32 = id.parse()?;

        let vector = vector
            .trim_matches(|c: char| c.is_whitespace() || c == '[' || c == ']')
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap())
            .collect();

        vectors.push((id, vector));
        assert_eq!(vectors[0].1.len(), vectors.last().unwrap().1.len());
    }

    let vector_len = vectors[0].1.len();
    eprintln!("The dimension of the vector is {}", vector_len);

    let mut writer = BufWriter::new(io::stdout());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut id = first_id;
    for _ in 0..(count.checked_div(2).unwrap()) {
        let mut iter = vectors.choose_multiple(&mut rng, 2);
        let a = iter.next().unwrap();
        let b = iter.next().unwrap();

        let (al, ar) = a.1.split_at(vector_len / 2);
        let (bl, br) = b.1.split_at(vector_len - al.len()); // even numbers

        let newa: Vec<f32> = al.iter().copied().chain(br.iter().copied()).collect();
        let newb: Vec<f32> = bl.iter().copied().chain(ar.iter().copied()).collect();

        writer.write_all(&id.to_be_bytes())?;
        newa.iter().try_for_each(|f| writer.write_all(&f.to_be_bytes()))?;

        writer.write_all(&(id + 1).to_be_bytes())?;
        newb.iter().try_for_each(|f| writer.write_all(&f.to_be_bytes()))?;

        id = id.checked_add(2).unwrap();
    }

    eprintln!("The last id generated was {}", id.saturating_sub(1));

    Ok(())
}
