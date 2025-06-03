use std::path::PathBuf;

use heed::RwTxn;
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::{PyIOError, PyRuntimeError}, prelude::*};

use crate::{distance, Database, ItemId, Writer};

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
    fn new(env: &heed::Env, wtxn: &mut RwTxn<'_>, name: Option<&str>, distance: DistanceType) -> heed::Result<DynDatabase> {
        match distance {
            DistanceType::Euclidean => {
                Ok(DynDatabase::Euclidean(env.create_database(wtxn, name)?))
            }
            DistanceType::Manhattan => {
                Ok(DynDatabase::Manhattan(env.create_database(wtxn, name)?))
            }
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
    fn new(path: PathBuf, name: Option<&str>, size: Option<usize>, distance: DistanceType) -> PyResult<PyDatabase> {
        let size = size.unwrap_or(TWENTY_HUNDRED_MIB);
        let env = unsafe { heed::EnvOpenOptions::new().map_size(size).open(path) }.map_err(h2py_err)?;

        let mut wtxn = env.write_txn().map_err(h2py_err)?;
        let db_impl = DynDatabase::new(&env, &mut wtxn, name, distance).map_err(h2py_err)?;
        Ok(PyDatabase(db_impl))
    }

    fn writer(&self, index: u16, dimensions: usize) -> PyWriter {
        match self.0 {
            DynDatabase::Euclidean(db) => PyWriter(DynWriter::Euclidean(Writer::new(db, index, dimensions))),
            DynDatabase::Manhattan(db) => PyWriter(DynWriter::Manhattan(Writer::new(db, index, dimensions))),
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
        let mut wtxn = get_txn();
        match &self.0 {
            DynWriter::Euclidean(writer) => writer.add_item(&mut wtxn, item, vector.as_slice()?).map_err(h2py_err),
            DynWriter::Manhattan(writer) => writer.add_item(&mut wtxn, item, vector.as_slice()?).map_err(h2py_err),
        }
    }
}

fn get_txn() -> heed::RwTxn<'static> {
    todo!("replace this with a Python context manager");
}

fn h2py_err<E: Into<crate::error::Error>>(e: E) -> PyErr {
    match e.into() {
        crate::Error::Heed(heed::Error::Io(e)) | crate::Error::Io(e) => PyIOError::new_err(e.to_string()),
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
