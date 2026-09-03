"""
Shared utility functions used across the project.
"""
import logging
import time
import functools
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import FIGURES_DIR


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def timer(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def save_figure(fig, filename: str, dpi: int = 150):
    """Save a matplotlib figure to the reports/figures directory."""
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return filepath


def set_plot_style():
    """Set consistent plotting style for the project."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def print_dataframe_summary(df: pd.DataFrame, name: str = "DataFrame"):
    """Print a comprehensive summary of a DataFrame."""
    print(f"\n{'='*60}")
    print(f"  {name} Summary")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  Duplicates: {df.duplicated().sum():,}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        print(f"\n  Missing Values ({len(missing_cols)} columns):")
        for col, count in missing_cols.head(10).items():
            pct = count / len(df) * 100
            print(f"    {col}: {count:,} ({pct:.1f}%)")
    else:
        print("  Missing Values: None")

    print(f"\n  Data Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"    {dtype}: {count}")
    print(f"{'='*60}\n")
