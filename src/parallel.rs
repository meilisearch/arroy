use core::slice;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::marker;
use std::sync::atomic::{AtomicU32, Ordering};

use heed::types::Bytes;
use heed::{BytesDecode, BytesEncode, RoTxn};
use memmap2::Mmap;
use roaring::RoaringBitmap;

use crate::internals::{KeyCodec, Leaf, NodeCodec};
use crate::key::{Prefix, PrefixCodec};
use crate::{Database, Distance, ItemId};

pub struct TmpNodes {
    file: BufWriter<File>,
    ids: Vec<u32>,
    bounds: Vec<usize>,
}

impl TmpNodes {
    pub fn new() -> heed::Result<TmpNodes> {
        let file = tempfile::tempfile().map(BufWriter::new)?;
        Ok(TmpNodes { file, ids: Vec::new(), bounds: vec![0] })
    }

    pub fn put<'a, DE: BytesEncode<'a>>(
        &mut self,
        item: u32,
        data: &'a DE::EItem,
    ) -> heed::Result<()> {
        let bytes = DE::bytes_encode(data).map_err(heed::Error::Encoding)?;
        self.file.write_all(&bytes)?;
        let last_bound = self.bounds.last().unwrap();
        self.bounds.push(last_bound + bytes.len());
        self.ids.push(item);
        Ok(())
    }

    pub fn into_reader(self) -> heed::Result<TmpReader> {
        let file = self.file.into_inner().unwrap(); // TODO fix
        Ok(TmpReader { mmap: unsafe { Mmap::map(&file)? }, ids: self.ids, bounds: self.bounds })
    }
}

pub struct TmpReader {
    mmap: Mmap,
    ids: Vec<u32>,
    bounds: Vec<usize>,
}

impl TmpReader {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, &[u8])> {
        self.ids.iter().zip(self.bounds.windows(2)).map(|(&id, bounds)| {
            let [start, end] = [bounds[0], bounds[1]];
            (id, &self.mmap[start..end])
        })
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct ConcurrentNodeIds(AtomicU32);

impl ConcurrentNodeIds {
    pub fn new(v: u32) -> ConcurrentNodeIds {
        ConcurrentNodeIds(AtomicU32::new(v))
    }

    pub fn next(&self) -> u32 {
        self.0.fetch_add(1, Ordering::SeqCst)
    }
}

pub struct ImmutableLeafs<'t, D> {
    leaf_ids: RoaringBitmap,
    constant_length: Option<usize>,
    offsets: Vec<*const u8>,
    _marker: marker::PhantomData<(&'t (), D)>,
}

impl<'t, D: Distance> ImmutableLeafs<'t, D> {
    pub fn new(
        rtxn: &'t RoTxn,
        database: Database<D>,
        index: u16,
    ) -> heed::Result<ImmutableLeafs<'t, D>> {
        let mut leaf_ids = RoaringBitmap::new();
        let mut constant_length = None;
        let mut offsets = Vec::new();

        let iter = database
            .remap_types::<PrefixCodec, Bytes>()
            .prefix_iter(rtxn, &Prefix::item(index))?
            .remap_key_type::<KeyCodec>();

        for result in iter {
            let (key, bytes) = result?;
            let item_id = key.node.unwrap_item();
            assert_eq!(*constant_length.get_or_insert(bytes.len()), bytes.len());
            assert!(leaf_ids.push(item_id));
            offsets.push(bytes.as_ptr());
        }

        Ok(ImmutableLeafs { leaf_ids, constant_length, offsets, _marker: marker::PhantomData })
    }

    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<Leaf<'t, D>>> {
        let len = match self.constant_length {
            Some(len) => len,
            None => return Ok(None),
        };
        let ptr = match self
            .leaf_ids
            .rank(item_id)
            .checked_sub(1)
            .and_then(|offset| self.offsets.get(offset as usize))
        {
            Some(ptr) => *ptr,
            None => return Ok(None),
        };
        let bytes = unsafe { slice::from_raw_parts(ptr, len) };
        NodeCodec::bytes_decode(bytes).map_err(heed::Error::Decoding).map(|node| node.leaf())
    }
}

unsafe impl<D> Sync for ImmutableLeafs<'_, D> {}
