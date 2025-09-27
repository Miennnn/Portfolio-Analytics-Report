# src/attribution.py
import pandas as pd

def brinson_allocation_selection(port_w, port_r, bench_w, bench_r):
    """
    Single-period Brinson-Fachler style attribution by segment (e.g., sector/region).

    Parameters
    ----------
    port_w : pd.Series
        Portfolio weights by segment (must sum ~1 across segments you care about).
    port_r : pd.Series
        Portfolio returns by segment (simple returns for the period).
    bench_w : pd.Series
        Benchmark weights by segment (sum ~1).
    bench_r : pd.Series
        Benchmark returns by segment (simple returns for the period).

    Returns
    -------
    pd.DataFrame
        Columns: Allocation, Selection, Interaction
        Index: segments + a 'Total' row (sum of columns).
    """
    # Align indexes across inputs
    idx = port_w.index.union(bench_w.index).union(port_r.index).union(bench_r.index)
    pw = port_w.reindex(idx, fill_value=0)
    bw = bench_w.reindex(idx, fill_value=0)
    pr = port_r.reindex(idx, fill_value=0)
    br = bench_r.reindex(idx, fill_value=0)

    # Benchmark total return (weighted)
    b_tot = (bw * br).sum()

    # Brinson–Fachler components (one-period)
    allocation  = (pw - bw) * (br - b_tot)
    selection   = bw * (pr - br)
    interaction = (pw - bw) * (pr - br)

    df = pd.DataFrame({
        "Allocation": allocation,
        "Selection": selection,
        "Interaction": interaction
    })

    # Add totals row
    df.loc["Total"] = df.sum()
    return df
