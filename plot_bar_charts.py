#!/usr/bin/env python
"""Plot wrapper: regenerate ensemble figures from genomic_ensemble.py — no training/GPU needed."""
import io
import os
import random
import sys

import numpy as np
import torch

# Fix Windows console encoding for Unicode characters (R², Δ, etc.)
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# PYTHONHASHSEED must be set at process launch: export PYTHONHASHSEED=42

from genomic_ensemble import (  # noqa: E402
    generate_bar_charts,
    generate_efficiency_plots,
    generate_scatter_plots,
)


def main():
    fig_dir = None
    if '--fig-dir' in sys.argv:
        idx = sys.argv.index('--fig-dir')
        if idx + 1 >= len(sys.argv):
            raise ValueError('--fig-dir requires a path')
        fig_dir = sys.argv[idx + 1]

    only_bar = '--bar-only' in sys.argv
    print("Regenerating figures via genomic_ensemble plotting wrappers...")
    generate_bar_charts(fig_dir=fig_dir)
    if not only_bar:
        generate_scatter_plots(fig_dir=fig_dir)
        generate_efficiency_plots(fig_dir=fig_dir)
    print("Done.")


if __name__ == '__main__':
    main()
