from pathlib import Path

import arroy
import numpy as np


def test_exports() -> None:
    assert arroy.__all__ == ["Database", "Writer"]


def test_create(tmp_path: Path) -> None:
    db = arroy.Database(tmp_path)
    with db.writer(0, 3) as writer:
        writer.add_item(0, np.array([0.1, 0.2, 0.3], dtype=np.float32))
