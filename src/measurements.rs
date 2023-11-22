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
    /// The number of calls to the internal `make_tree` function.
    pub calls_to_make_tree: usize,
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
        calls_to_make_tree: MAKE_TREES.swap(0, Ordering::SeqCst),
    }
}

macro_rules! increment {
    ($static_name:ident, $function_name:ident) => {
        #[cfg(feature = "measurements")]
        pub static $static_name: AtomicUsize = AtomicUsize::new(0);

        #[cfg(feature = "measurements")]
        pub fn $function_name() {
            $static_name.fetch_add(1, Ordering::SeqCst);
        }

        #[cfg(not(feature = "measurements"))]
        pub fn $function_name() {}
    };
}

increment!(ALIGNED_VECTOR, increment_aligned_vectors);
increment!(UNALIGNED_VECTOR, increment_unaligned_vectors);
increment!(MAKE_TREES, increment_make_trees);
