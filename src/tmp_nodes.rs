use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU32, Ordering};

use heed::BytesEncode;
use memmap2::Mmap;

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
