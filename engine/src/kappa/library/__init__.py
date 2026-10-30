"""Ready protocols for common paradigms, to start from instead of a drafted one.

Each is a plain ``AssayProtocol`` JSON file. Their arena corners are ``null``: the
corners depend on the camera, so a person marks them on a frame before confirming.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from kappa.protocol import AssayProtocol


def names() -> list[str]:
    return sorted(p.name.removesuffix(".json") for p in files(__name__).iterdir() if p.name.endswith(".json"))


def load(name: str) -> dict[str, Any]:
    if name not in names():
        raise KeyError(f"no library protocol {name}; there are: {', '.join(names())}")
    return json.loads(files(__name__).joinpath(f"{name}.json").read_text(encoding="utf-8"))


def protocol(name: str) -> AssayProtocol:
    return AssayProtocol.model_validate(load(name))


def catalogue() -> list[dict[str, Any]]:
    """Every library protocol with its title, hash and what it still needs before it can run."""
    entries = []
    for name in names():
        p = protocol(name)
        entries.append(
            {
                "name": name,
                "title": p.title,
                "sha256": p.content_hash(),
                "needs": ["arena.corners_px"] if p.arena.corners_px is None else [],
                "notes": p.notes,
                "protocol": p.model_dump(mode="json"),
            }
        )
    return entries
