"""Content-addressed storage for every artifact a measurement depends on.

Model outputs are not reproducible run to run, so Kappa never re-asks a model for
something it already has: each raw output is stored under its sha256 and the
manifest of a run lists those hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "blobs").mkdir(parents=True, exist_ok=True)
        (root / "index").mkdir(parents=True, exist_ok=True)

    def _blob(self, digest: str) -> Path:
        return self.root / "blobs" / digest[:2] / digest

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        path = self._blob(digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return digest

    def put_text(self, text: str) -> str:
        return self.put(text.encode("utf-8"))

    def put_json(self, value: Any) -> str:
        return self.put_text(json.dumps(value, sort_keys=True, separators=(",", ":")))

    def get(self, digest: str) -> bytes:
        return self._blob(digest).read_bytes()

    def get_text(self, digest: str) -> str:
        return self.get(digest).decode("utf-8")

    def get_json(self, digest: str) -> Any:
        return json.loads(self.get_text(digest))

    def has(self, digest: str) -> bool:
        return self._blob(digest).exists()

    def remember(self, key: str, digest: str) -> None:
        """Map a request key (e.g. the hash of a request body) to the output it produced."""
        (self.root / "index" / hashlib.sha256(key.encode()).hexdigest()).write_text(digest)

    def recall(self, key: str) -> str | None:
        path = self.root / "index" / hashlib.sha256(key.encode()).hexdigest()
        return path.read_text().strip() if path.exists() else None
