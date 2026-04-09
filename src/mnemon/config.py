"""Configuration loading and defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "base_dir": "~/.mnemon",
    "bind_host": "127.0.0.1",
    "bind_port": 7474,
    "default_resolver": "strict",
    "chroma_collection": "mnemon_chunks",
    "l1_max_items": 15,
    "l1_max_chars": 3200,
    "l1_item_max_chars": 200,
    "chunk_target_tokens": 180,
    "chunk_overlap_chars": 100,
    "domain_map": {},
}


@dataclass
class MnemonConfig:
    base_dir: Path
    bind_host: str = "127.0.0.1"
    bind_port: int = 7474
    default_resolver: str = "strict"
    chroma_collection: str = "mnemon_chunks"
    l1_max_items: int = 15
    l1_max_chars: int = 3200
    l1_item_max_chars: int = 200
    chunk_target_tokens: int = 180
    chunk_overlap_chars: int = 100
    domain_map: dict[str, list[str]] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.base_dir / "knowledge_graph.db"

    @property
    def chroma_dir(self) -> Path:
        return self.base_dir / "chroma"

    @property
    def canonical_dir(self) -> Path:
        return self.base_dir / "canonical"

    @property
    def identity_path(self) -> Path:
        return self.base_dir / "identity.txt"

    @property
    def l1_cache_path(self) -> Path:
        return self.base_dir / "l1_cache.txt"

    @property
    def saves_log_path(self) -> Path:
        return self.base_dir / "saves.log"

    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


def load_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> MnemonConfig:
    """Load config from JSON file, applying defaults and overrides."""
    merged = dict(DEFAULT_CONFIG)

    # MNEMON_BASE_DIR env var takes precedence over default (for containers)
    env_base = os.environ.get("MNEMON_BASE_DIR")
    if env_base:
        merged["base_dir"] = env_base

    if config_path is None:
        config_path = Path(merged["base_dir"]).expanduser() / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            merged.update(json.load(f))

    if overrides:
        merged.update(overrides)

    base_dir = Path(merged.pop("base_dir")).expanduser()

    return MnemonConfig(base_dir=base_dir, **{
        k: v for k, v in merged.items() if k in MnemonConfig.__dataclass_fields__
    })
