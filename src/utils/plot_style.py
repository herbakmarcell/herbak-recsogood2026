"""
utils/plot_style.py
====================
Shared light "report" theme for all matplotlib figures in this project.

Replaces the ad-hoc dark-navy rcParams blocks that used to be copy-pasted
into every plotting script (and, incidentally, the ``matplotlib.cm.get_cmap``
calls that crash on matplotlib >= 3.9 - use :func:`get_cmap` below instead).
"""

import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BG_COLOR      = "#ffffff"
PANEL_COLOR   = "#f7f7fa"
TEXT_COLOR    = "#1a1a2e"
GRID_COLOR    = "#d8d8e0"
MUTED_COLOR   = "#5a5a6e"

PALETTE = [
    "#4C63FF", "#E0527A", "#2FA787", "#E08A2E", "#2C7FB8",
    "#A64BC7", "#1AA6A6", "#D9622B", "#6FA341", "#C9A227",
]

MODEL_COLORS = {
    "BPR":      "#4C63FF",
    "MultiDAE": "#E08A2E",
    "EASE":     "#2FA787",
    "ELSA":     "#E0527A",
    "LightGCN": "#2C7FB8",
    "NeuMF":    "#A64BC7",
}


def apply_style():
    """Apply the shared light report theme to matplotlib's rcParams."""
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "figure.dpi":         150,
        "figure.facecolor":   BG_COLOR,
        "axes.facecolor":     BG_COLOR,
        "savefig.facecolor":  BG_COLOR,
        "text.color":         TEXT_COLOR,
        "axes.labelcolor":    TEXT_COLOR,
        "axes.edgecolor":     GRID_COLOR,
        "xtick.color":        MUTED_COLOR,
        "ytick.color":        MUTED_COLOR,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": False,
        "axes.grid":          True,
        "grid.color":         GRID_COLOR,
        "grid.linewidth":     0.8,
        "grid.alpha":         0.6,
        "grid.linestyle":     "--",
    })


def get_cmap(name: str):
    """
    matplotlib>=3.9-safe replacement for the removed ``cm.get_cmap`` /
    ``plt.get_cmap`` shortcut used throughout the old plotting scripts.
    """
    return matplotlib.colormaps[name]


def model_color(model_name: str, fallback_index: int = 0):
    """Look up a model's colour, falling back to the shared PALETTE."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    return PALETTE[fallback_index % len(PALETTE)]


def configure_stdout():
    """
    Force UTF-8 stdout so the box-drawing / unicode characters printed by the
    plotting scripts never crash on a Windows terminal using a legacy codepage
    (e.g. cp1250). No-op where ``reconfigure`` is unavailable. This replaces the
    identical ``if hasattr(sys.stdout, "reconfigure"): ...`` block that used to
    be copy-pasted at the top of every script.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


MARKER_POOL = ["o", "s", "^", "D", "*", "P", "X", "v", "<", ">", "h"]


def strategy_markers(strategies):
    """Assign a stable marker to each strategy from :data:`MARKER_POOL`."""
    return {s: MARKER_POOL[i % len(MARKER_POOL)] for i, s in enumerate(strategies)}


def short_label(s: str) -> str:
    """
    Canonical short strategy label (two-line for the long strategy names, plus
    the ``Coreset-* -> CS-*`` abbreviations) shared by the scatter/bar scripts.
    Formerly duplicated verbatim as a local ``short()`` in several scripts.
    """
    return (s.replace("User-based", "User-\nbased")
             .replace("Item-based", "Item-\nbased")
             .replace("User-temporal", "User-\ntemporal")
             .replace("Global-temporal", "Global-\ntemporal")
             .replace("Coreset-Leverage", "CS-Leverage")
             .replace("Coreset-Cluster",  "CS-Cluster"))


def _log_tick_formatter(v, _pos):
    """Readable label for a log-axis tick, e.g. ``2×10⁻⁴`` or ``10⁻³``."""
    if v <= 0:
        return ""
    exp  = int(np.floor(np.log10(v) + 1e-9))
    mant = v / 10.0 ** exp
    if abs(mant - 1.0) < 1e-6:
        return r"$10^{%d}$" % exp
    return r"$%g{\times}10^{%d}$" % (round(mant, 1), exp)


def set_log_ticks(ax, which="x", subs=(1.0, 2.0, 3.0, 5.0)):
    """
    Give a log-scaled axis readable, multi-decade ticks.

    Matplotlib's default ``LogLocator`` only labels integer powers of ten, so a
    dataset spanning less than one decade (typical for the tiny kWh values here)
    shows a single, uninformative ``10⁻⁴`` tick. Placing major ticks at *subs*
    (1,2,3,5)·10ⁿ and formatting them compactly restores a legible scale.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(mticker.LogLocator(base=10, subs=subs, numticks=15))
    axis.set_minor_locator(mticker.LogLocator(base=10, subs="auto", numticks=30))
    axis.set_minor_formatter(mticker.NullFormatter())
    axis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))


def robust_limits(values, pad=0.08, k=2.5):
    """
    Axis limits that frame the bulk of *values*, ignoring extreme outliers.

    Some green-tradeoff points (e.g. a strategy that *increases* energy several-
    fold) sit orders of magnitude away from the rest and, left unchecked, squash
    every interesting point into a sliver. Limits are derived from the IQR
    (median ± ``k``·IQR) rather than min/max; callers clamp+annotate any point
    that falls outside so no information is lost. Returns ``(lo, hi)``.
    """
    flat = []
    for x in values:                       # tolerate scalars, arrays, or Series
        flat.extend(np.asarray(x, dtype=float).ravel().tolist())
    v = np.asarray([x for x in flat if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (-1.0, 1.0)
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        lo_fence, hi_fence = q1 - k * iqr, q3 + k * iqr
        inliers = v[(v >= lo_fence) & (v <= hi_fence)]
        if inliers.size == 0:
            inliers = v
        lo, hi = float(inliers.min()), float(inliers.max())
    else:
        lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or (abs(hi) or 1.0)
    return (lo - pad * span, hi + pad * span)


def spread_labels(ys, gap):
    """
    Nudge a set of label y-positions so consecutive ones differ by at least
    *gap*, preserving input order. Used to stop the pinned off-scale annotations
    on the green-tradeoff plots from stacking on top of each other when several
    strategies land at the same axis edge. Returns a list aligned with *ys*.
    """
    ys = list(ys)
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        i, prev = order[k], order[k - 1]
        if out[i] - out[prev] < gap:
            out[i] = out[prev] + gap
    return out
