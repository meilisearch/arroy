#[cfg(feature = "measurements")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg(feature = "measurements")]
pub struct Measurements {
    /// Represents the number of time we were able to cast a slice of bytes
    /// into a slice of something else and avoid copying into a `Vec`.
    pub aligned_slices_read: usize,
    /// Represents the number of time we were forced to allocate memory into
    /// a `Vec` to make sure it is correctly aligned and can be used safely.
    pub unaligned_slices_read: usize,
}

/// Returns the measurements done during the set of operations measured.
///
/// Note that each measurement is reset one by one, it is not atomic.
/// You must make sure not to use the library while calling this function.
#[cfg(feature = "measurements")]
pub fn get_and_reset_measurements() -> Measurements {
    Measurements {
        aligned_slices_read: ALIGNED_VECTOR.swap(0, Ordering::SeqCst),
        unaligned_slices_read: UNALIGNED_VECTOR.swap(0, Ordering::SeqCst),
    }
}

// -----

#[cfg(feature = "measurements")]
pub static ALIGNED_VECTOR: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "measurements")]
pub fn increment_aligned_vectors() {
    ALIGNED_VECTOR.fetch_add(1, Ordering::SeqCst);
}

#[cfg(not(feature = "measurements"))]
pub fn increment_aligned_vectors() {}

// -----

#[cfg(feature = "measurements")]
pub static UNALIGNED_VECTOR: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "measurements")]
pub fn increment_unaligned_vectors() {
    UNALIGNED_VECTOR.fetch_add(1, Ordering::SeqCst);
}

#[cfg(not(feature = "measurements"))]
pub fn increment_unaligned_vectors() {}
