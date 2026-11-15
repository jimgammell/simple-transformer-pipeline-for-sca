"""Small, pickle-free artifact identity records shared by evaluation entrypoints."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(checkpoint, dataset, settings):
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "hyper_parameters"
    ]
    mtd = saved.get("mtd_kwargs", {})
    checkpoint_settings = dict(
        preprocessing=saved.get("preprocessing"),
        leakage_model=saved.get("leakage_model"),
        num_labels=saved.get("num_labels"),
        attack_count=mtd.get("attack_count"),
        traces_per_attack=mtd.get("traces_per_attack"),
    )
    root = Path(__file__).resolve().parents[2]
    # Hash retained implementation, not machine-specific source paths or Git HEAD
    # (which does not identify an uncommitted working tree).
    sources = sorted((root / "src").rglob("*.py")) + sorted(
        (root / "experiments").rglob("*.py")
    )
    source_hash = hashlib.sha256()
    for path in sources:
        source_hash.update(str(path.relative_to(root)).encode())
        source_hash.update(bytes.fromhex(sha256_file(path)))
    return dict(
        format_version=1,
        checkpoint_sha256=sha256_file(checkpoint),
        source_sha256=source_hash.hexdigest(),
        dataset=dataset,
        normalization_source="checkpoint persistent statistics buffers",
        checkpoint_settings=checkpoint_settings,
        settings=settings,
    )


def metadata_json(metadata):
    return np.array(json.dumps(metadata, sort_keys=True))


def cache_matches(path, identity, overwrite=False):
    if overwrite or not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(data["metadata"].item())
        if metadata["identity"] == identity:
            return True
    except (ValueError, KeyError, TypeError, OSError):
        pass
    raise ValueError(
        f"Cached artifact {path} has missing or mismatched identity; use --overwrite to replace it."
    )


def seed_runs(root):
    """Find the complete five-seed reporting cohort."""
    runs = sorted(Path(root).glob("seed=*"))
    if {run.name for run in runs} != {f"seed={seed}" for seed in range(5)}:
        raise ValueError(f"{root}: expected exactly seed=0 through seed=4")
    return runs


def load_attack(run):
    """Load saved key ranks in the one-based convention used by sensitivity plots."""
    with np.load(Path(run) / "attack_metrics.npz", allow_pickle=False) as saved:
        ranks = saved["ranks_zero_based"]
    if (
        ranks.ndim != 3
        or not all(ranks.shape)
        or ranks.shape[2] != 16
        or not np.issubdtype(ranks.dtype, np.integer)
        or np.any(ranks < 0)
        or np.any(ranks > 255)
    ):
        raise ValueError(f"{run}: expected integer zero-based ranks [A,T,16] in 0..255")
    return ranks.astype(np.int16) + 1


def summarize_ranks(ranks):
    ranks = np.asarray(ranks)
    if (
        ranks.ndim != 3
        or not all(ranks.shape)
        or not np.isfinite(ranks).all()
        or not ((ranks >= 1) & (ranks <= 256) & (ranks == np.floor(ranks))).all()
    ):
        raise ValueError(
            "Expected finite integer one-based ranks in 1..256 with shape [A,T,B]."
        )
    return dict(
        ranks_zero_based=(ranks - 1).astype(np.uint8),
        guessing_entropy=ranks.mean(axis=0).T,
        success_rate=(ranks == 1).mean(axis=0).T,
        full_key_success_rate=(ranks == 1).all(axis=2).mean(axis=0),
    )
