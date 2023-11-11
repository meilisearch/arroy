#! /bin/bash

# Runs the classic and heed-based version of Annoy and compare them side-by-side.
# Very useful when you make sure to log the same things in both programs.

set -v

cargo run --bin classic > classic.out
cargo run --bin heed > heed.out

diff --side-by-side classic.out heed.out
