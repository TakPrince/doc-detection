from __future__ import annotations

import hashlib
import random


def derive_seed(base_seed: int, *parts: object) -> int:
    if not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    payload = "\x00".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def make_rng(base_seed: int, *parts: object) -> random.Random:
    return random.Random(derive_seed(base_seed, *parts))


def seed_everything(seed: int) -> None:
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed % (2**32))


__all__ = ["derive_seed", "make_rng", "seed_everything"]
