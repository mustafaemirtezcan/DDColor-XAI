from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset: str
    image_path: Path
    mask_path: Path
    label: str
    category_id: int | None = None


def read_manifest(path: str | Path, limit: int | None = None) -> Iterator[Sample]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            record = json.loads(line)
            yield Sample(
                sample_id=str(record["id"]),
                dataset=str(record["dataset"]),
                image_path=Path(record["image"]).expanduser().resolve(),
                mask_path=Path(record["mask"]).expanduser().resolve(),
                label=str(record.get("label", "unknown")),
                category_id=(
                    int(record["category_id"])
                    if record.get("category_id") is not None
                    else None
                ),
            )
