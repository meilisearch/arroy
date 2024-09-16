use binary_quantized_test::binary_quantized::{from_slice_non_optimized, to_vec_non_optimized};
use insta::{assert_debug_snapshot, assert_snapshot};
use proptest::collection::vec;
use proptest::prelude::*;

use super::*;
use crate::internals::UnalignedVectorCodec;

#[test]
fn test_from_slice() {
    let original = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9];
    let vector = BinaryQuantized::from_slice(&original);

    #[allow(clippy::format_collect)]
    let internal = vector.as_bytes().iter().map(|b| format!("{b:08b}\n")).collect::<String>();
    assert_snapshot!(internal, @r###"
        10101011
        00000000
        00000000
        00000000
        00000000
        00000000
        00000000
        00000000
        "###);
}

#[test]
fn test_to_vec_iter() {
    let original = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9];
    let vector = BinaryQuantized::from_slice(&original);
    let iter_vec: Vec<_> = BinaryQuantized::iter(&vector).take(original.len()).collect();
    assert_debug_snapshot!(iter_vec, @r###"
        [
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
        ]
        "###);
    let mut vec_vec: Vec<_> = BinaryQuantized::to_vec(&vector);
    vec_vec.truncate(original.len());
    assert_debug_snapshot!(vec_vec, @r###"
        [
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
        ]
        "###);

    assert_eq!(vec_vec, iter_vec);
}

proptest! {
    #[test]
    fn from_slice_simd_vs_non_optimized(
        original in vec(-50f32..=50.2, 0..516)
    ){
        let vector = BinaryQuantized::from_slice(&original);
        let iter_vec: Vec<_> = to_vec_non_optimized(&vector);
        let vec_vec: Vec<_> = BinaryQuantized::to_vec(&vector);

        assert_eq!(vec_vec, iter_vec);
    }

    #[test]
    fn to_vec_simd_vs_non_optimized(
        original in vec(-50f32..=50.2, 0..516)
    ){
        let vector1 = BinaryQuantized::from_slice(&original);
        let vector2 = from_slice_non_optimized(&original);

        assert_eq!(vector1.as_bytes(), &vector2);
    }
}
