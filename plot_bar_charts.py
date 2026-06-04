#!/usr/bin/env python
"""Plot wrapper: regenerate ensemble figures from genomic_ensemble.py — no training needed."""
import io
import random
import sys

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - plotting wrapper can run without torch
    torch = None

RANDOM_SEED = 42


def set_global_seed(seed=RANDOM_SEED):
    """Set process-local RNGs for reproducible plotting/wrapper behavior."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(RANDOM_SEED)
# PYTHONHASHSEED must be set at process launch: export PYTHONHASHSEED=42

# Fix Windows console encoding for Unicode characters (R², Δ, etc.)
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

from genomic_ensemble import generate_all_plots  # noqa: E402


def main():
    fig_dir = None
    if '--fig-dir' in sys.argv:
        idx = sys.argv.index('--fig-dir')
        if idx + 1 >= len(sys.argv):
            raise ValueError('--fig-dir requires a path')
        fig_dir = sys.argv[idx + 1]

    only_bar = '--bar-only' in sys.argv
    print("Regenerating figures via genomic_ensemble plotting wrappers...")
    ok = generate_all_plots(fig_dir=fig_dir, include_scatter=not only_bar,
                            include_efficiency=not only_bar)
    if not ok:
        print("Plot generation finished with failures.")
        sys.exit(2)
    print("Done.")


if __name__ == '__main__':
    main()
