use std::{
    collections::{HashMap, HashSet},
    fmt,
    hash::{BuildHasherDefault, Hasher},
    marker::PhantomData,
};

/// A `HashMap` with integer keys, using `NoHashHasher`.
pub type IntMap<K, V> = HashMap<K, V, BuildNoHashHasher<K>>;

/// A `HashSet` of integers, using `NoHash`.
pub type IntSet<T> = HashSet<T, BuildNoHashHasher<T>>;

/// An alias for `BuildHasherDefault` for use with `NoHash`.
pub type BuildNoHashHasher<T> = BuildHasherDefault<NoHash<T>>;

/// `std::hash::Hasher` implementation which maps input value as its hash output.
pub struct NoHash<T>(u64, PhantomData<T>);

impl<T> fmt::Debug for NoHash<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("NoHash").field(&self.0).finish()
    }
}

impl<T> Default for NoHash<T> {
    fn default() -> Self {
        NoHash(0, PhantomData)
    }
}

pub trait IsEnabled {}

impl IsEnabled for u8 {}
impl IsEnabled for u16 {}
impl IsEnabled for u32 {}
impl IsEnabled for u64 {}
impl IsEnabled for usize {}
impl IsEnabled for i8 {}
impl IsEnabled for i16 {}
impl IsEnabled for i32 {}
impl IsEnabled for i64 {}
impl IsEnabled for isize {}

impl<T: IsEnabled> Hasher for NoHash<T> {
    fn write(&mut self, _: &[u8]) {
        panic!("Invalid use of NoHash")
    }

    fn write_u8(&mut self, n: u8) {
        self.0 = u64::from(n)
    }
    fn write_u16(&mut self, n: u16) {
        self.0 = u64::from(n)
    }
    fn write_u32(&mut self, n: u32) {
        self.0 = u64::from(n)
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n
    }
    fn write_usize(&mut self, n: usize) {
        self.0 = n as u64
    }

    fn write_i8(&mut self, n: i8) {
        self.0 = n as u64
    }
    fn write_i16(&mut self, n: i16) {
        self.0 = n as u64
    }
    fn write_i32(&mut self, n: i32) {
        self.0 = n as u64
    }
    fn write_i64(&mut self, n: i64) {
        self.0 = n as u64
    }
    fn write_isize(&mut self, n: isize) {
        self.0 = n as u64
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok() {
        let mut h1 = NoHash::<u8>::default();
        h1.write_u8(42);
        assert_eq!(42, h1.finish());

        let mut h2 = NoHash::<u16>::default();
        h2.write_u16(42);
        assert_eq!(42, h2.finish());

        let mut h3 = NoHash::<u32>::default();
        h3.write_u32(42);
        assert_eq!(42, h3.finish());

        let mut h4 = NoHash::<u64>::default();
        h4.write_u64(42);
        assert_eq!(42, h4.finish());

        let mut h5 = NoHash::<usize>::default();
        h5.write_usize(42);
        assert_eq!(42, h5.finish());

        let mut h6 = NoHash::<i8>::default();
        h6.write_i8(42);
        assert_eq!(42, h6.finish());

        let mut h7 = NoHash::<i16>::default();
        h7.write_i16(42);
        assert_eq!(42, h7.finish());

        let mut h8 = NoHash::<i32>::default();
        h8.write_i32(42);
        assert_eq!(42, h8.finish());

        let mut h9 = NoHash::<i64>::default();
        h9.write_i64(42);
        assert_eq!(42, h9.finish());

        let mut h10 = NoHash::<isize>::default();
        h10.write_isize(42);
        assert_eq!(42, h10.finish())
    }
}
