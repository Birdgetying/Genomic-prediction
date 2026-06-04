#!/usr/bin/env python
"""Plot wrapper: regenerate ensemble figures from genomic_ensemble.py — no training needed."""
import io
import sys

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
    generate_all_plots(fig_dir=fig_dir, include_scatter=not only_bar,
                       include_efficiency=not only_bar)
    print("Done.")


if __name__ == '__main__':
    main()
