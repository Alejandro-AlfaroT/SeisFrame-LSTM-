"""Filesystem locations. Everything stays on this machine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CORTEX_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    db_path: Path
    backup_dir: Path
    frontend_dist: Path

    @classmethod
    def from_env(cls, db_path: str | Path | None = None) -> "Settings":
        data_dir = Path(os.environ.get("CORTEX_DATA_DIR", CORTEX_ROOT / "data"))
        db = Path(db_path or os.environ.get("CORTEX_DB", data_dir / "cortex.db"))
        return cls(
            db_path=db,
            backup_dir=Path(os.environ.get("CORTEX_BACKUP_DIR", db.parent / "backups")),
            frontend_dist=CORTEX_ROOT / "frontend" / "dist",
        )
