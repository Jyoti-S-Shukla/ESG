"""
Page-adaptive table extraction for borderless, multi-column PDF reports.

Why this exists: pdfplumber's built-in `page.extract_tables()` uses its own
row/column clustering, which on this specific document (no visible
gridlines, varying line-heights between the small "Sl. No" numerals and the
main paragraph text) was dropping the Sl. No column and misaligning rows --
verified directly against the real uploaded PDF, not assumed.

Approach instead:
    1. Detect column boundaries per page from whitespace *gutters* -- runs
       of x-coordinates with zero character coverage across the page.
       This is computed fresh per page, so it adapts automatically if
       column widths differ between the Table 1 section (fewer columns)
       and the Table 2 section (Sl.No + BRSR + GRI + Remarks).
    2. Cluster words into visual rows by y-position (`top`), independent
       of pdfplumber's table engine.
    3. Assign each word to a column bucket by its x-center, and join words
       within a (row, column) cell in reading order.

This gives one grid row per PDF *line*, not per logical record (a BRSR
requirement's GRI text wraps across many lines/rows) -- merging wrapped
lines into logical records happens downstream (in 03/04), keyed off the
Sl. No / BRSR-ID column being non-empty as the start-of-record signal.
"""

import numpy as np


def detect_column_gutters(page, margin=35, min_gutter_width=6):
    """Return x-coordinate column boundaries for a page, as a list of
    boundary positions [left_edge, gutter1, gutter2, ..., right_edge]."""
    chars = page.chars
    width = int(page.width) + 2
    coverage = np.zeros(width)
    for c in chars:
        x0, x1 = max(0, int(c["x0"])), min(width, int(c["x1"]) + 1)
        coverage[x0:x1] += 1

    gutters = []
    in_gap, gap_start = False, 0
    for x in range(margin, width - margin):
        if coverage[x] == 0 and not in_gap:
            in_gap, gap_start = True, x
        elif coverage[x] > 0 and in_gap:
            in_gap = False
            if x - gap_start >= min_gutter_width:
                gutters.append((gap_start + x) // 2)

    return [margin] + gutters + [width - margin]


def words_to_grid(page, col_bounds, row_tolerance=3):
    """Reconstruct a table grid from a page's words, given column
    boundaries. Returns a list of rows, each a list of per-column strings
    (one grid row per visual line of text, not per logical record)."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    row_clusters = []
    current_row, current_top = [], None
    for w in words:
        if current_top is None or abs(w["top"] - current_top) <= row_tolerance:
            current_row.append(w)
            current_top = w["top"] if current_top is None else current_top
        else:
            row_clusters.append(current_row)
            current_row, current_top = [w], w["top"]
    if current_row:
        row_clusters.append(current_row)

    n_cols = len(col_bounds) - 1
    grid = []
    for row_words in row_clusters:
        cells = [[] for _ in range(n_cols)]
        for w in row_words:
            x_center = (w["x0"] + w["x1"]) / 2
            for i in range(n_cols):
                if col_bounds[i] <= x_center < col_bounds[i + 1]:
                    cells[i].append(w)
                    break
        cell_texts = [
            " ".join(w["text"] for w in sorted(cell, key=lambda w: w["x0"]))
            for cell in cells
        ]
        grid.append(cell_texts)
    return grid


def extract_page_grid(page, margin=35, min_gutter_width=6, row_tolerance=3):
    """Convenience wrapper: detect gutters and build the grid in one call."""
    col_bounds = detect_column_gutters(page, margin=margin, min_gutter_width=min_gutter_width)
    grid = words_to_grid(page, col_bounds, row_tolerance=row_tolerance)
    return grid, col_bounds
