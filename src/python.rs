use std::{path::PathBuf, sync::LazyLock};

// TODO: replace with std::sync::Mutex once MutexGuard::map is stable.
use numpy::PyReadonlyArray1;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
// TODO: replace with std::sync::OnceLock once get_or_try_init is stable.
use once_cell::sync::OnceCell as OnceLock;
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError},
    prelude::*,
};

use crate::{distance, Database, ItemId, Writer};

static ENV: OnceLock<heed::Env> = OnceLock::new();
static RW_TXN: LazyLock<Mutex<Option<heed::RwTxn<'static>>>> = LazyLock::new(|| Mutex::new(None));

const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;

#[pyclass]
#[derive(Debug, Clone, Copy)]
enum DistanceType {
    Euclidean,
    Manhattan,
}

#[derive(Debug, Clone, Copy)]
enum DynDatabase {
    Euclidean(Database<distance::Euclidean>),
    Manhattan(Database<distance::Manhattan>),
}

impl DynDatabase {
    fn new(
        env: &heed::Env,
        wtxn: &mut heed::RwTxn<'_>,
        name: Option<&str>,
        distance: DistanceType,
    ) -> heed::Result<DynDatabase> {
        match distance {
            DistanceType::Euclidean => Ok(DynDatabase::Euclidean(env.create_database(wtxn, name)?)),
            DistanceType::Manhattan => Ok(DynDatabase::Manhattan(env.create_database(wtxn, name)?)),
        }
    }
}

#[pyclass(name = "Database")]
#[derive(Debug, Clone)]
struct PyDatabase(DynDatabase);

#[pymethods]
impl PyDatabase {
    #[new]
    #[pyo3(signature = (path, name = None, size = None, distance = DistanceType::Euclidean))]
    fn new(
        path: PathBuf,
        name: Option<&str>,
        size: Option<usize>,
        distance: DistanceType,
    ) -> PyResult<PyDatabase> {
        let size = size.unwrap_or(TWENTY_HUNDRED_MIB);
        // TODO: allow one per path, allow destroying and recreating, etc.
        let env = ENV
            .get_or_try_init(|| unsafe { heed::EnvOpenOptions::new().map_size(size).open(path) })
            .map_err(h2py_err)?;

        let mut wtxn = get_rw_txn()?;
        let db_impl = DynDatabase::new(env, &mut wtxn, name, distance).map_err(h2py_err)?;
        Ok(PyDatabase(db_impl))
    }

    fn writer(&self, index: u16, dimensions: usize) -> PyWriter {
        match self.0 {
            DynDatabase::Euclidean(db) => {
                PyWriter(DynWriter::Euclidean(Writer::new(db, index, dimensions)))
            }
            DynDatabase::Manhattan(db) => {
                PyWriter(DynWriter::Manhattan(Writer::new(db, index, dimensions)))
            }
        }
    }

    #[staticmethod]
    fn commit_rw_txn() -> PyResult<()> {
        if let Some(wtxn) = RW_TXN.lock().take() {
            wtxn.commit().map_err(h2py_err)?;
        }
        Ok(())
    }

    #[staticmethod]
    fn abort_rw_txn() {
        if let Some(wtxn) = RW_TXN.lock().take() {
            wtxn.abort();
        }
    }
}

#[derive(Debug)]
enum DynWriter {
    Euclidean(Writer<distance::Euclidean>),
    Manhattan(Writer<distance::Manhattan>),
}

#[pyclass(name = "Writer")]
struct PyWriter(DynWriter);

#[pymethods]
impl PyWriter {
    fn add_item(&mut self, item: ItemId, vector: PyReadonlyArray1<f32>) -> PyResult<()> {
        let mut wtxn = get_rw_txn()?;
        match &self.0 {
            DynWriter::Euclidean(writer) => {
                writer.add_item(&mut wtxn, item, vector.as_slice()?).map_err(h2py_err)
            }
            DynWriter::Manhattan(writer) => {
                writer.add_item(&mut wtxn, item, vector.as_slice()?).map_err(h2py_err)
            }
        }
    }
}

/// Get the current transaction or start it.
fn get_rw_txn<'a>() -> PyResult<MappedMutexGuard<'a, heed::RwTxn<'static>>> {
    let mut maybe_txn = RW_TXN.lock();
    if maybe_txn.is_none() {
        let env = ENV.get().ok_or_else(|| PyRuntimeError::new_err("No environment"))?;
        let rw_txn = env.write_txn().map_err(h2py_err)?;
        *maybe_txn = Some(rw_txn);
    };
    // unwrapping since if the value was None when we got the lock, we just set it.
    Ok(MutexGuard::map(maybe_txn, |txn| txn.as_mut().unwrap()))
}

fn h2py_err<E: Into<crate::error::Error>>(e: E) -> PyErr {
    match e.into() {
        crate::Error::Heed(heed::Error::Io(e)) | crate::Error::Io(e) => {
            PyIOError::new_err(e.to_string())
        }
        e => PyRuntimeError::new_err(e.to_string()),
    }
}

#[pyo3::pymodule]
#[pyo3(name = "arroy")]
pub fn pymodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyWriter>()?;
    Ok(())
}
