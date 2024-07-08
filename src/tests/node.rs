use insta::{assert_debug_snapshot, assert_snapshot};

use crate::unaligned_vector::{BinaryQuantized, UnalignedVector};

#[test]
fn unaligned_f32_vec() {
    let original: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let bytes: Vec<u8> = original.iter().flat_map(|f| f.to_ne_bytes()).collect();

    let unaligned_owned_from_f32 = UnalignedVector::<f32>::from_vec(original.clone());
    assert_eq!(bytes, unaligned_owned_from_f32.as_bytes());

    let unchecked_unaligned_owned_from_bytes = UnalignedVector::<f32>::from_bytes_unchecked(&bytes);
    assert_eq!(bytes, unchecked_unaligned_owned_from_bytes.as_bytes());

    let unaligned_owned_from_bytes = UnalignedVector::<f32>::from_bytes(&bytes).unwrap();
    assert_eq!(bytes, unaligned_owned_from_bytes.as_bytes());
}

#[test]
fn unaligned_binary_quantized_smol() {
    let original: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0, 5.0];

    let unaligned = UnalignedVector::<BinaryQuantized>::from_slice(&original);
    let s = unaligned.as_bytes().iter().map(|byte| format!("{byte:08b}\n")).collect::<String>();
    assert_snapshot!(s, @r###"
    00011010
    00000000
    00000000
    00000000
    00000000
    00000000
    00000000
    00000000
    "###);

    let deser: Vec<_> = unaligned.iter().collect();
    assert_debug_snapshot!(deser[0..original.len()], @r###"
    [
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
    ]
    "###);
}

#[test]
fn unaligned_binary_quantized_large() {
    let original: Vec<f32> = (0..100)
        .map(|n| if n % 3 == 0 || n % 5 == 0 { n as f32 * -1.0 } else { n as f32 })
        .collect();

    // Two numbers should be used
    let unaligned = UnalignedVector::<BinaryQuantized>::from_slice(&original);
    let s = unaligned.as_bytes().iter().map(|byte| format!("{byte:08b}\n")).collect::<String>();
    assert_snapshot!(s, @r###"
    10010110
    01101001
    11001011
    10110100
    01100101
    11011010
    00110010
    01101101
    10011001
    10110110
    01001100
    01011011
    00000110
    00000000
    00000000
    00000000
    "###);

    let deser: Vec<_> = unaligned.to_vec();
    assert_snapshot!(format!("{:?}", &deser[0..original.len()]),
@"[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]");
    for (orig, deser) in original.iter().zip(&deser) {
        if orig.is_sign_positive() {
            assert_eq!(deser, &1.0);
        } else {
            assert_eq!(deser, &0.0);
        }
    }
}
