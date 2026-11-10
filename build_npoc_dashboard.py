#!/usr/bin/env python3
"""
Build the NPOC initial-training dashboard (single self-contained HTML file).

Usage
-----
    python build_npoc_dashboard.py path/to/training_monitoring.csv
    python build_npoc_dashboard.py data.csv -o docs/index.html --asof "10 Nov 2026"
    python build_npoc_dashboard.py data.csv --embed-js      # embed plotly.js (needs `pip install plotly`)

Privacy
-------
The HTML contains ONLY aggregated numbers (counts, regional means, test
statistics). Facility names, participant names, e-mails and row-level scores are
never written to the output. After writing, the script scans the HTML for any
facility / participant / e-mail value found in the CSV and aborts if one is
present. Keep the CSV itself out of the repository (see .gitignore note in the
chat reply).

Libraries: pandas, numpy, scipy (stats). Charts are Plotly figure specs
(the same JSON `plotly.graph_objects` produces) rendered by plotly.js.
"""
import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Settings you may want to change
# --------------------------------------------------------------------------- #
PLOTLYJS_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
MIN_N = 5                         # minimum paired records before a t-test is reported
ZERO_PRETEST_IS_MISSING = True    # a pretest of exactly 0 is treated as "not taken"
REGION_ORDER = ["NCR", "CAR", "I", "II", "III", "IV", "MMRP", "V", "VI", "VII",
                "VIII", "IX", "X", "XI", "XII", "XIII", "NIR", "BARMM"]

METRICS = [  # (short key, axis label, CSV column)
    ("pre", "Pretest", "PRETEST_PERCENT"),
    ("post", "Posttest", "POSTTEST_PERCENT"),
    ("comp", "Competency<br>assessment", "COMPETENCY_PERCENT"),
    ("pl", "Pluslife<br>assessment", "PLUSLIFE_PERCENT"),
]

INK, TEAL, BLUE, CORAL, AMBER, SLATE = "#16233B", "#0E7C7B", "#3F7CAC", "#C8553D", "#D99A2B", "#8A97AB"
GRID = "#E3E8EE"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def num(x):
    """float or None (JSON-safe)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else round(x, 4)


def to_num(series):
    return pd.to_numeric(series.str.replace("%", "", regex=False).str.strip(), errors="coerce")


def region_sort_key(r):
    return (REGION_ORDER.index(r), r) if r in REGION_ORDER else (len(REGION_ORDER), r)


def hake_category(g):
    if g is None or (isinstance(g, float) and math.isnan(g)):
        return "n/a"
    return "high" if g >= 0.7 else "medium" if g >= 0.3 else "low"


def p_text(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "p = n/a"
    return "p &lt; 0.001" if p < 0.001 else f"p = {p:.3f}"


def stars(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " (ns)"


# --------------------------------------------------------------------------- #
# Load + clean
# --------------------------------------------------------------------------- #
def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda s: s.str.strip())
    df["REGION"] = df["REGION"].str.upper().replace({"": "UNSPECIFIED"})
    df["SELECTION"] = df["SELECTION"].str.upper()
    df["OVERALL_STATUS"] = df["OVERALL_STATUS"].str.upper()
    df["HELPER_STATUS"] = df["HELPER_STATUS"].str.upper()
    for key, _label, col in METRICS:
        df["s_" + key] = to_num(df[col])
    n_zero_pre = int((df["s_pre"] == 0).sum())
    if ZERO_PRETEST_IS_MISSING:
        df.loc[df["s_pre"] == 0, "s_pre"] = np.nan
    return df, n_zero_pre


# --------------------------------------------------------------------------- #
# Section 1 + 3: coverage numbers
# --------------------------------------------------------------------------- #
def coverage(df):
    main = df[df["SELECTION"] == "MAIN"]
    alt = df[df["SELECTION"] == "ALTERNATE"]

    # Cross-check the sheet's own HELPER_STATUS against a recomputation
    alt_trained_for = set(alt.loc[alt["OVERALL_STATUS"] == "TRAINED", "ALTERNATE_FOR"])
    recomputed = np.where(main["OVERALL_STATUS"] == "TRAINED", "MAIN TRAINED",
                          np.where(main["FACILITY"].isin(alt_trained_for), "ALTERNATE TRAINED", "NEITHER TRAINED"))
    mismatch = int((recomputed != main["HELPER_STATUS"].values).sum())
    if mismatch:
        print(f"WARNING: {mismatch} main rows where HELPER_STATUS differs from recomputed status "
              f"(using HELPER_STATUS).", file=sys.stderr)

    regions = sorted(df["REGION"].unique(), key=region_sort_key)
    cov = pd.DataFrame(index=regions)
    cov["main_total"] = main.groupby("REGION").size()
    cov["main_trained"] = main[main["HELPER_STATUS"] == "MAIN TRAINED"].groupby("REGION").size()
    cov["alt_covered"] = main[main["HELPER_STATUS"] == "ALTERNATE TRAINED"].groupby("REGION").size()
    cov["neither"] = main[main["HELPER_STATUS"] == "NEITHER TRAINED"].groupby("REGION").size()
    cov["alt_total"] = alt.groupby("REGION").size()
    cov = cov.fillna(0).astype(int)
    cov["covered"] = cov["main_trained"] + cov["alt_covered"]
    cov["pct"] = np.where(cov["main_total"] > 0, 100 * cov["covered"] / cov["main_total"].replace(0, np.nan), np.nan)

    inc = df[df["OVERALL_STATUS"].str.startswith("NOT TRAINED FOR")].copy()
    inc["kind"] = np.where(inc["OVERALL_STATUS"].str.contains("ONLINE"), "online", "f2f")
    inc_ct = {(s, k): int(((inc["SELECTION"] == s) & (inc["kind"] == k)).sum())
              for s in ("MAIN", "ALTERNATE") for k in ("online", "f2f")}

    totals = dict(
        main_total=int(len(main)),
        main_trained=int((main["HELPER_STATUS"] == "MAIN TRAINED").sum()),
        alt_covered=int((main["HELPER_STATUS"] == "ALTERNATE TRAINED").sum()),
        neither=int((main["HELPER_STATUS"] == "NEITHER TRAINED").sum()),
        alt_listed=int(len(alt)),
        alt_trained=int((alt["OVERALL_STATUS"] == "TRAINED").sum()),
        incomplete=int(len(inc)),
        neither_incomplete=int(((main["HELPER_STATUS"] == "NEITHER TRAINED")
                                & main["OVERALL_STATUS"].str.startswith("NOT TRAINED FOR")).sum()),
        n_regions=len(regions),
    )
    totals["covered"] = totals["main_trained"] + totals["alt_covered"]
    totals["pct"] = 100 * totals["covered"] / totals["main_total"]
    return cov, totals, inc_ct, regions


# --------------------------------------------------------------------------- #
# Section 2: performance + pre/post statistics
# --------------------------------------------------------------------------- #
def paired_stats(sub):
    d = sub[["s_pre", "s_post"]].dropna()
    n = len(d)
    out = dict(n=n, pre=None, post=None, diff=None, t=None, df=None, p=None, dz=None, g=None)
    if n == 0:
        return out
    pre, post = d["s_pre"].mean(), d["s_post"].mean()
    out.update(pre=pre, post=post, diff=post - pre)
    out["g"] = (post - pre) / (100 - pre) if pre < 100 else None      # Hake: group-average gain
    if n >= MIN_N:
        diff = d["s_post"] - d["s_pre"]
        sd = diff.std(ddof=1)
        if sd > 0:
            t, p = stats.ttest_rel(d["s_post"], d["s_pre"])
            out.update(t=float(t), p=float(p), df=n - 1, dz=float(diff.mean() / sd))
    return out


def performance(df, regions):
    groups = {"All regions": df}
    for r in regions:
        groups[r] = df[df["REGION"] == r]
    res = {}
    for name, sub in groups.items():
        m = {}
        for key, _label, _col in METRICS:
            s = sub["s_" + key].dropna()
            m[key] = dict(mean=s.mean() if len(s) else None,
                          sd=s.std(ddof=1) if len(s) > 1 else None, n=int(len(s)))
        res[name] = dict(metrics=m, paired=paired_stats(sub))
    return res


# --------------------------------------------------------------------------- #
# Plotly figure specs (plain dicts == plotly.graph_objects .to_dict())
# --------------------------------------------------------------------------- #
def base_layout(**kw):
    lay = dict(font=dict(family=FONT, size=13, color=INK), paper_bgcolor="rgba(0,0,0,0)",
               plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=60, r=24, t=30, b=50),
               hoverlabel=dict(font=dict(family=FONT, size=13)))
    lay.update(kw)
    return lay


def fig_status_bar(t):
    segs = [("Trained at main site", t["main_trained"], TEAL),
            ("Covered by alternate site", t["alt_covered"], BLUE),
            ("Neither trained", t["neither"], CORAL)]
    data = [dict(type="bar", orientation="h", y=["Main sites"], x=[n], name=name, text=[str(n)],
                 textposition="inside", insidetextanchor="middle",
                 textfont=dict(color="white", size=18), marker=dict(color=c),
                 hovertemplate=f"{name}<br>%{{x}} of {t['main_total']} main sites"
                               f" ({100 * n / t['main_total']:.1f}%)<extra></extra>")
            for name, n, c in segs]
    layout = base_layout(barmode="stack", height=250, margin=dict(l=10, r=10, t=10, b=90),
                         xaxis=dict(range=[0, t["main_total"]], showgrid=False, zeroline=False,
                                    showticklabels=False),
                         yaxis=dict(showticklabels=False, showgrid=False),
                         legend=dict(orientation="h", x=0, y=-0.2, yanchor="top"), bargap=0.15)
    return dict(data=data, layout=layout)


def fig_incomplete(inc_ct):
    cats = ["Missing online training", "Missing face-to-face training"]
    data = []
    for sel, label, color in (("MAIN", "Main sites", INK), ("ALTERNATE", "Alternate sites", AMBER)):
        vals = [inc_ct[(sel, "online")], inc_ct[(sel, "f2f")]]
        data.append(dict(type="bar", orientation="h", y=cats, x=vals, name=label,
                         text=[str(v) if v else "" for v in vals], textposition="inside",
                         insidetextanchor="middle", textfont=dict(color="white", size=15),
                         marker=dict(color=color),
                         hovertemplate=f"{label}<br>%{{y}}: %{{x}}<extra></extra>"))
    layout = base_layout(barmode="stack", height=250, margin=dict(l=10, r=10, t=10, b=90),
                         xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, dtick=1, rangemode="tozero"),
                         yaxis=dict(autorange="reversed", automargin=True),
                         legend=dict(orientation="h", x=0, y=-0.3, yanchor="top"), bargap=0.35)
    return dict(data=data, layout=layout)


def panel_text(label, s):
    head = f"<b>{label}: pretest to posttest</b>"
    if s["n"] == 0:
        return head + "<br><br>No paired pretest and posttest records."
    lines = [head, "",
             f"Paired records: <b>{s['n']}</b>",
             f"Mean pretest: <b>{s['pre']:.1f}%</b>",
             f"Mean posttest: <b>{s['post']:.1f}%</b>",
             f"Mean gain: <b>{s['diff']:+.1f} points</b>", ""]
    if s["p"] is None:
        lines += [f"<b>Paired t-test</b>", f"Not reported (fewer than {MIN_N} paired records)"]
    else:
        lines += ["<b>Paired t-test</b>", f"t({s['df']}) = {s['t']:.2f}, {p_text(s['p'])}",
                  f"Cohen's dz = {s['dz']:.2f}"]
    lines += ["", "<b>Hake's normalized gain</b>"]
    if s["g"] is None:
        lines.append("Not defined (pretest mean is 100%)")
    else:
        lines.append(f"g = {s['g']:.2f} ({hake_category(s['g'])} gain)")
    return "<br>".join(lines)


def fig_explorer(perf, regions):
    options = ["All regions"] + regions
    labels = [m[1] for m in METRICS]
    nat = perf["All regions"]["metrics"]
    nat_vals = [num(nat[k]["mean"]) for k, _l, _c in METRICS]
    data = [dict(type="bar", name="All regions (comparison)", x=labels, y=nat_vals,
                 marker=dict(color="rgba(138,151,171,0.45)"), visible=False,
                 hovertemplate="%{x}<br>All regions: %{y:.1f}%<extra></extra>")]
    for opt in options:
        m = perf[opt]["metrics"]
        ys = [num(m[k]["mean"]) for k, _l, _c in METRICS]
        data.append(dict(type="bar", name=opt, x=labels, y=ys,
                         text=[f"{v:.1f}%" if v is not None else "" for v in ys],
                         textposition="outside", cliponaxis=False, marker=dict(color=TEAL),
                         customdata=[[m[k]["n"], num(m[k]["sd"])] for k, _l, _c in METRICS],
                         hovertemplate="%{x}<br>Mean: %{y:.1f}%<br>n = %{customdata[0]}"
                                       "<br>SD = %{customdata[1]:.1f}<extra>" + opt + "</extra>",
                         visible=(opt == "All regions")))

    def ann(opt):
        return dict(xref="paper", yref="paper", x=0.64, y=0.97, xanchor="left", yanchor="top",
                    align="left", showarrow=False, text=panel_text(opt, perf[opt]["paired"]),
                    bgcolor="#F1F5F9", bordercolor=GRID, borderwidth=1, borderpad=16,
                    font=dict(size=14, color=INK))

    buttons = []
    for i, opt in enumerate(options):
        vis = [i != 0] + [j == i for j in range(len(options))]
        buttons.append(dict(label=opt, method="update",
                            args=[{"visible": vis}, {"annotations": [ann(opt)]}]))
    layout = base_layout(
        height=470, barmode="group", bargap=0.35, bargroupgap=0.08,
        margin=dict(l=60, r=24, t=80, b=100),
        xaxis=dict(domain=[0, 0.58], showgrid=False, fixedrange=True),
        yaxis=dict(range=[0, 112], ticksuffix="%", gridcolor=GRID, zeroline=False,
                   title=dict(text="Mean score")),
        legend=dict(orientation="h", x=0, y=-0.2, yanchor="top"),
        annotations=[ann("All regions")],
        updatemenus=[dict(type="dropdown", direction="down", active=0, x=0, xanchor="left",
                          y=1.2, yanchor="top", buttons=buttons, bgcolor="white", bordercolor="#B8C2D0",
                          font=dict(size=14), showactive=True, pad=dict(l=0, r=0, t=0, b=0))])
    return dict(data=data, layout=layout)


def fig_heatmap(perf, regions):
    rows = ["All regions"] + regions
    labels = [m[1] for m in METRICS]
    z = [[num(perf[r]["metrics"][k]["mean"]) for k, _l, _c in METRICS] for r in rows]
    flat = [v for row in z for v in row if v is not None]
    zmin = max(0, math.floor(min(flat) / 5) * 5) if flat else 0
    text = [[f"{v:.1f}" if v is not None else "" for v in row] for row in z]
    data = [dict(type="heatmap", x=labels, y=rows, z=z, text=text, texttemplate="%{text}",
                 zmin=zmin, zmax=100, xgap=3, ygap=3, showscale=True,
                 colorscale=[[0, "#F5ECD7"], [0.5, "#8CC5BC"], [1, TEAL]],
                 colorbar=dict(title=dict(text="Mean %"), thickness=12, len=0.6),
                 hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>")]
    layout = base_layout(height=max(420, 26 * len(rows) + 110), margin=dict(l=90, r=10, t=10, b=60),
                         xaxis=dict(side="bottom", fixedrange=True),
                         yaxis=dict(autorange="reversed", fixedrange=True, type="category"))
    return dict(data=data, layout=layout)


def fig_hake(perf, regions):
    pts = [(r, perf[r]["paired"]) for r in regions
           if perf[r]["paired"]["g"] is not None and perf[r]["paired"]["p"] is not None]
    pts.sort(key=lambda kv: kv[1]["g"], reverse=True)
    col = {"high": TEAL, "medium": AMBER, "low": CORAL}
    xs = [r for r, _ in pts]
    gs = [num(s["g"]) for _, s in pts]
    data = [dict(type="bar", x=xs, y=gs, showlegend=False,
                 marker=dict(color=[col[hake_category(s["g"])] for _, s in pts]),
                 text=[f"{s['g']:.2f}{stars(s['p'])}" for _, s in pts], textposition="outside",
                 cliponaxis=False,
                 customdata=[[s["n"], num(s["pre"]), num(s["post"]), num(s["p"])] for _, s in pts],
                 hovertemplate="%{x}<br>Hake's g = %{y:.2f}<br>Paired n = %{customdata[0]}"
                               "<br>Pretest %{customdata[1]:.1f}% to posttest %{customdata[2]:.1f}%"
                               "<br>Paired t-test p = %{customdata[3]:.4g}<extra></extra>")]
    for name, c in (("High gain (g ≥ 0.7)", TEAL), ("Medium gain (0.3 ≤ g < 0.7)", AMBER), ("Low gain (g < 0.3)", CORAL)):
        data.append(dict(type="bar", x=[None], y=[None], name=name, marker=dict(color=c), hoverinfo="skip"))
    shapes, annotations = [], []
    g_all = perf["All regions"]["paired"]["g"]
    if g_all is not None:
        shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=g_all, y1=g_all,
                           line=dict(color=INK, width=1.5, dash="dash")))
        annotations.append(dict(xref="paper", x=1, xanchor="right", yref="y", y=g_all, yanchor="bottom",
                                text=f"All regions: g = {g_all:.2f}", showarrow=False,
                                font=dict(size=12, color=INK)))
    for y in (0.3, 0.7):
        shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=y, y1=y,
                           line=dict(color=SLATE, width=1, dash="dot")))
    ymax = max([g for g in gs if g is not None] + [1.0])
    layout = base_layout(height=430, barmode="overlay", margin=dict(l=60, r=24, t=30, b=100),
                         xaxis=dict(type="category", showgrid=False, fixedrange=True),
                         yaxis=dict(range=[min(0, min(gs) - 0.05) if gs else 0, ymax + 0.15],
                                    gridcolor=GRID, zeroline=True, zerolinecolor=SLATE,
                                    title=dict(text="Hake's normalized gain (g)")),
                         legend=dict(orientation="h", x=0, y=-0.18, yanchor="top"),
                         shapes=shapes, annotations=annotations, bargap=0.3)
    return dict(data=data, layout=layout)


def fig_coverage(cov, totals):
    regions = list(cov.index)
    pct = [num(v) for v in cov["pct"]]
    data = [
        dict(type="bar", name="Total main sites", x=regions, y=cov["main_total"].tolist(),
             marker=dict(color=INK), yaxis="y",
             hovertemplate="%{x}<br>Main sites: %{y}<extra></extra>"),
        dict(type="bar", name="Main sites covered", x=regions, y=cov["covered"].tolist(),
             marker=dict(color=TEAL), yaxis="y",
             customdata=list(zip(cov["main_trained"].tolist(), cov["alt_covered"].tolist())),
             hovertemplate="%{x}<br>Covered: %{y}<br>Trained at main: %{customdata[0]}"
                           "<br>Via alternate: %{customdata[1]}<extra></extra>"),
        dict(type="bar", name="Total alternate sites", x=regions, y=cov["alt_total"].tolist(),
             marker=dict(color=AMBER), yaxis="y",
             hovertemplate="%{x}<br>Alternate sites: %{y}<extra></extra>"),
        dict(type="scatter", mode="markers+text", name="% coverage of main sites", x=regions, y=pct,
             yaxis="y2", marker=dict(color=CORAL, size=11),
             text=[f"{v:.0f}%" if v is not None else "" for v in pct], textposition="top center",
             textfont=dict(size=12, color=CORAL), cliponaxis=False,
             hovertemplate="%{x}<br>Coverage: %{y:.1f}%<extra></extra>"),
    ]
    layout = base_layout(
        height=660, barmode="group", bargap=0.25, margin=dict(l=70, r=24, t=20, b=110),
        xaxis=dict(type="category", showgrid=False, anchor="y", fixedrange=True),
        yaxis=dict(domain=[0, 0.64], gridcolor=GRID, zeroline=False, title=dict(text="Number of sites")),
        yaxis2=dict(domain=[0.74, 1], range=[0, 118], tickvals=[0, 50, 100], ticksuffix="%", anchor="x",
                    gridcolor=GRID, zeroline=False, title=dict(text="Coverage")),
        legend=dict(orientation="h", x=0, y=-0.14, yanchor="top"),
        shapes=[dict(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=totals["pct"], y1=totals["pct"],
                     line=dict(color=CORAL, width=1, dash="dot"))],
        annotations=[dict(xref="paper", x=1, xanchor="right", yref="y2", y=totals["pct"], yanchor="top",
                          text=f"All regions: {totals['pct']:.1f}%", showarrow=False,
                          font=dict(size=12, color=CORAL))])
    return dict(data=data, layout=layout)


def fig_composition(cov):
    c = cov.sort_values("pct", ascending=True)
    ylabels = [f"{r}  ({v:.0f}%)" for r, v in zip(c.index, c["pct"])]
    series = [("Trained at main site", "main_trained", TEAL), ("Covered by alternate site", "alt_covered", BLUE),
              ("Not covered", "neither", CORAL)]
    data = [dict(type="bar", orientation="h", y=ylabels, x=c[col].tolist(), name=name,
                 marker=dict(color=color), text=[str(v) if v else "" for v in c[col]],
                 textposition="inside", insidetextanchor="middle", textfont=dict(color="white", size=12),
                 hovertemplate="%{y}<br>" + name + ": %{x} main sites<extra></extra>")
            for name, col, color in series]
    layout = base_layout(height=max(420, 28 * len(c) + 110), barmode="stack", barnorm="percent", bargap=0.25,
                         margin=dict(l=110, r=24, t=10, b=100),
                         xaxis=dict(ticksuffix="%", gridcolor=GRID, zeroline=False, fixedrange=True),
                         yaxis=dict(type="category", automargin=True, fixedrange=True),
                         legend=dict(orientation="h", x=0, y=-0.12, yanchor="top"))
    return dict(data=data, layout=layout)


def fig_table(cov, totals):
    rows = list(cov.index) + ["All regions"]
    def col(name, total):
        return cov[name].astype(int).tolist() + [total]
    pct = [f"{v:.1f}%" for v in cov["pct"]] + [f"{totals['pct']:.1f}%"]
    cells = [rows, col("main_total", totals["main_total"]), col("main_trained", totals["main_trained"]),
             col("alt_covered", totals["alt_covered"]), col("covered", totals["covered"]),
             col("neither", totals["neither"]), col("alt_total", totals["alt_listed"]), pct]
    last = len(rows) - 1
    fill = [["#FFFFFF" if i != last else "#E8F1F1" for i in range(len(rows))]] * len(cells)
    data = [dict(type="table", columnwidth=[1.1, 1, 1, 1.1, 1, 1, 1, 1.1],
                 header=dict(values=["Region", "Main sites", "Trained at main", "Covered by alternate",
                                     "Total covered", "Not covered", "Alternate sites", "% coverage"],
                             fill=dict(color=INK), font=dict(color="white", size=13, family=FONT),
                             align="center", height=40, line=dict(color=INK)),
                 cells=dict(values=cells, fill=dict(color=fill), align="center", height=30,
                            font=dict(size=13, color=INK, family=FONT), line=dict(color=GRID)))]
    layout = base_layout(height=40 + 30 * len(rows) + 30, margin=dict(l=0, r=0, t=0, b=0))
    return dict(data=data, layout=layout)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NPOC initial training dashboard</title>
__PLOTLY_TAG__
<style>
  :root {
    --ink:#16233B; --muted:#5B6B82; --line:#E3E8EE; --bg:#F5F7FA; --card:#FFFFFF;
    --teal:#0E7C7B; --blue:#3F7CAC; --coral:#C8553D; --amber:#D99A2B;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
         line-height:1.5; -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1180px; margin:0 auto; padding:0 24px; }
  header { background:var(--ink); color:#fff; padding:44px 0 36px; }
  header h1 { margin:0 0 8px; font-size:clamp(26px,4vw,40px); line-height:1.15; letter-spacing:-0.02em; font-weight:700; }
  header p { margin:0; max-width:62ch; color:#C9D3E3; font-size:16px; }
  header .stamp { margin-top:14px; font-size:13px; color:#98A7BF; }
  nav { background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }
  nav .wrap { display:flex; gap:26px; overflow-x:auto; }
  nav a { display:block; padding:14px 0; color:var(--muted); text-decoration:none; font-size:14px; font-weight:600;
          white-space:nowrap; border-bottom:2px solid transparent; }
  nav a:hover, nav a:focus-visible { color:var(--ink); border-bottom-color:var(--teal); outline:none; }
  section { padding:44px 0 8px; }
  section > .wrap > h2 { margin:0 0 6px; font-size:26px; letter-spacing:-0.01em; }
  section > .wrap > .lede { margin:0 0 24px; color:var(--muted); max-width:70ch; }
  .kpis { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:14px; }
  .kpi { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px 18px; border-top:4px solid var(--c); }
  .kpi .num { font-size:34px; font-weight:700; letter-spacing:-0.02em; line-height:1.1; }
  .kpi .lbl { font-weight:600; margin-top:4px; }
  .kpi .sub { color:var(--muted); font-size:13px; margin-top:4px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:20px 22px 12px; margin-bottom:18px; }
  .card h3 { margin:0 0 4px; font-size:18px; }
  .card .desc { margin:0 0 12px; color:var(--muted); font-size:14px; max-width:80ch; }
  .two { display:grid; grid-template-columns:1.25fr 1fr; gap:18px; }
  .two .card { margin-bottom:0; }
  .chart { width:100%; }
  .scroll-x { overflow-x:auto; }
  .explorer { min-width:760px; }
  .notes { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:20px 24px; margin:44px 0 0; color:var(--muted); font-size:14px; }
  .notes h2 { margin:0 0 8px; font-size:18px; color:var(--ink); }
  .notes p { margin:0 0 10px; max-width:90ch; }
  footer { padding:28px 0 48px; color:var(--muted); font-size:13px; }
  @media (max-width:980px) { .kpis { grid-template-columns:repeat(2,1fr); } .two { grid-template-columns:1fr; } .two .card { margin-bottom:18px; } }
  @media (max-width:480px) { .wrap { padding:0 14px; } .kpis { grid-template-columns:1fr; } .card { padding:14px 12px 8px; } }
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>NPOC initial training: coverage and performance</h1>
    <p>How many main sites are covered by trained staff, how participants performed, and where coverage still needs attention, by region.</p>
    <div class="stamp">Aggregated figures only. Generated __ASOF__.</div>
  </div>
</header>
<nav aria-label="Sections">
  <div class="wrap">
    <a href="#summary">Summary</a>
    <a href="#performance">Training performance</a>
    <a href="#coverage">Regional coverage</a>
    <a href="#notes">Notes and definitions</a>
  </div>
</nav>

<section id="summary">
  <div class="wrap">
    <h2>Summary</h2>
    <p class="lede">A main site counts as covered when its own staff completed training, or when an alternate site assigned to it did.</p>
    <div class="kpis">__KPIS__</div>
    <div class="two">
      <div class="card">
        <h3>How the __MAIN_TOTAL__ main sites are covered</h3>
        <p class="desc">Each main site falls in exactly one group.</p>
        <div id="fig-status" class="chart"></div>
      </div>
      <div class="card">
        <h3>Incomplete training</h3>
        <p class="desc">Participants who finished only one of the two training components.</p>
        <div id="fig-incomplete" class="chart"></div>
      </div>
    </div>
  </div>
</section>

<section id="performance">
  <div class="wrap">
    <h2>Training performance</h2>
    <p class="lede">Average scores for the pretest, posttest, competency assessment and Pluslife assessment. Pick a region to compare it with all regions and to see its pretest-to-posttest statistics.</p>
    <div class="card">
      <h3>Regional explorer</h3>
      <p class="desc">Use the menu at the top left of the chart. Grey bars show the all-regions average for comparison. Hover a bar for the number of records and the standard deviation.</p>
      <div class="scroll-x"><div id="fig-explorer" class="chart explorer"></div></div>
    </div>
    <div class="card">
      <h3>All regions at a glance</h3>
      <p class="desc">Mean score (%) per assessment. Darker cells are higher.</p>
      <div id="fig-heatmap" class="chart"></div>
    </div>
    <div class="card">
      <h3>Learning gain by region (Hake's g)</h3>
      <p class="desc">Share of the possible improvement that participants achieved between pretest and posttest. Stars show the paired t-test result: * p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001.</p>
      <div id="fig-hake" class="chart"></div>
    </div>
  </div>
</section>

<section id="coverage">
  <div class="wrap">
    <h2>Regional coverage</h2>
    <p class="lede">Coverage is the share of a region's main sites that are trained at the main site or through an alternate.</p>
    <div class="card">
      <h3>Main sites, covered sites and alternate sites</h3>
      <p class="desc">Bars show counts. The dots above show each region's percent coverage.</p>
      <div id="fig-coverage" class="chart"></div>
    </div>
    <div class="card">
      <h3>How each region's coverage was reached</h3>
      <p class="desc">Regions are sorted from lowest coverage (top) to highest. Percent coverage is in brackets after the region name.</p>
      <div id="fig-composition" class="chart"></div>
    </div>
    <div class="card">
      <h3>Coverage table</h3>
      <div id="fig-table" class="chart"></div>
    </div>
  </div>
</section>

<section id="notes">
  <div class="wrap">
    <div class="notes">
      <h2>Notes and definitions</h2>
      __NOTES__
    </div>
  </div>
</section>
<footer><div class="wrap">Built with Python (pandas, SciPy) and Plotly.</div></footer>

<script type="application/json" id="figure-specs">__SPECS__</script>
<script>
(function () {
  var specs = JSON.parse(document.getElementById("figure-specs").textContent);
  var config = { responsive: true, displaylogo: false, displayModeBar: "hover",
                 modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };
  var ids = { "fig-status": "status", "fig-incomplete": "incomplete", "fig-explorer": "explorer",
              "fig-heatmap": "heatmap", "fig-hake": "hake", "fig-coverage": "coverage",
              "fig-composition": "composition", "fig-table": "table" };
  if (typeof Plotly === "undefined") {
    document.querySelectorAll(".chart").forEach(function (el) {
      el.innerHTML = "<p style='color:#C8553D'>Charts could not load. Check your internet connection (plotly.js is loaded from a CDN) or rebuild with --embed-js.</p>";
    });
    return;
  }
  Object.keys(ids).forEach(function (domId) {
    var s = specs[ids[domId]];
    Plotly.newPlot(domId, s.data, s.layout, config);
  });
})();
</script>
</body>
</html>
"""


def kpi(num_, label, sub, color):
    return (f'<div class="kpi" style="--c:{color}"><div class="num">{num_}</div>'
            f'<div class="lbl">{label}</div><div class="sub">{sub}</div></div>')


def build_html(specs, t, inc_ct, n_zero_pre, perf, asof, plotly_tag):
    inc_main = inc_ct[("MAIN", "online")] + inc_ct[("MAIN", "f2f")]
    inc_alt = inc_ct[("ALTERNATE", "online")] + inc_ct[("ALTERNATE", "f2f")]
    inc_online = inc_ct[("MAIN", "online")] + inc_ct[("ALTERNATE", "online")]
    inc_f2f = inc_ct[("MAIN", "f2f")] + inc_ct[("ALTERNATE", "f2f")]
    kpis = "".join([
        kpi(f"{t['covered']}", "Sites trained (main + alternate)",
            f"{t['pct']:.1f}% of {t['main_total']} main sites are covered", TEAL),
        kpi(f"{t['main_trained']}", "Trained at the main site",
            f"{100 * t['main_trained'] / t['main_total']:.1f}% of main sites", INK),
        kpi(f"{t['alt_covered']}", "Covered by an alternate",
            f"{t['alt_trained']} of {t['alt_listed']} listed alternates trained", BLUE),
        kpi(f"{t['neither']}", "Neither trained",
            f"No trained main or alternate staff ({t['neither_incomplete']} have partial training)", CORAL),
        kpi(f"{t['incomplete']}", "Incomplete training",
            f"{inc_online} missing online, {inc_f2f} missing face-to-face ({inc_main} main, {inc_alt} alternate)", AMBER),
    ])
    g = perf["All regions"]["paired"]
    if g["p"] is not None:
        stat_line = (f"Across all regions, {g['n']} paired records show a mean gain of {g['diff']:+.1f} points "
                     f"(pretest {g['pre']:.1f}% to posttest {g['post']:.1f}%), paired t({g['df']}) = {g['t']:.2f}, "
                     f"{p_text(g['p'])}, Cohen's dz = {g['dz']:.2f}, Hake's g = {g['g']:.2f}.")
    else:
        stat_line = ""
    zero_note = (f"{n_zero_pre} pretest scores of exactly 0% were treated as not taken and left out of every "
                 f"pretest average, t-test and gain calculation, because a pretest of 0 paired with a near-perfect "
                 f"posttest would inflate the gain. Set ZERO_PRETEST_IS_MISSING = False in the build script to include them."
                 if ZERO_PRETEST_IS_MISSING and n_zero_pre else "")
    notes = "".join(f"<p>{x}</p>" for x in [
        "<b>Main site covered.</b> A main site is covered when the main site itself is trained, or when it is not "
        "trained but at least one alternate site assigned to it is. Percent coverage = covered main sites divided by total main sites.",
        f"<b>Alternate sites.</b> {t['alt_listed']} alternate sites are listed; {t['alt_trained']} are trained. "
        f"{t['alt_trained'] - t['alt_covered']} trained alternates are not counted as coverage because their main site was already trained or another alternate already covered it.",
        "<b>Incomplete training.</b> A participant trained in only one of the online and face-to-face components. "
        "Main sites in this group are counted as neither trained.",
        "<b>Averages.</b> Means use every valid score for that assessment, so the number of records can differ by assessment. "
        "Records marked not found or no score are excluded.",
        "<b>Paired t-test.</b> Two-tailed, pretest versus posttest, using only records with both scores. "
        f"Not reported when a region has fewer than {MIN_N} paired records. Cohen's dz is the mean gain divided by the standard deviation of the gains.",
        "<b>Hake's normalized gain.</b> g = (mean posttest − mean pretest) / (100 − mean pretest), computed on group means. "
        "Common reading: g &lt; 0.3 low, 0.3 to &lt; 0.7 medium, ≥ 0.7 high. Because pretest scores are already high, small absolute gains can still give a high g.",
        zero_note, stat_line,
        "<b>Privacy.</b> This page contains only aggregated counts and statistics. No facility names, participant names, e-mail addresses or individual scores are included.",
    ] if x)
    specs_json = json.dumps(specs, allow_nan=False, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (TEMPLATE.replace("__PLOTLY_TAG__", plotly_tag).replace("__ASOF__", asof)
            .replace("__KPIS__", kpis).replace("__MAIN_TOTAL__", str(t["main_total"]))
            .replace("__NOTES__", notes).replace("__SPECS__", specs_json))
    return html


def privacy_check(html, df):
    """Abort if any facility / participant / e-mail string from the CSV appears in the HTML."""
    haystack = html.lower()
    leaks = []
    for col in ("FACILITY", "PARTICIPANT", "PARTICIPANT_HELPER1", "PARTICIPANT_HELPER2", "EMAIL"):
        if col not in df.columns:
            continue
        for v in df[col].dropna().unique():
            v = v.strip()
            if len(v) >= 6 and v.upper() not in {"NOT FOUND", "NO SCORE"} and v.lower() in haystack:
                leaks.append((col, v))
    if leaks:
        print("PRIVACY CHECK FAILED: identifying values found in output:", file=sys.stderr)
        for col, v in leaks[:10]:
            print(f"  {col}: {v}", file=sys.stderr)
        sys.exit(1)
    print("Privacy check passed: no facility, participant or e-mail values appear in the HTML.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="training monitoring CSV")
    ap.add_argument("-o", "--output", default="npoc_dashboard.html")
    ap.add_argument("--asof", default=dt.date.today().strftime("%d %b %Y"), help="date shown in the header")
    ap.add_argument("--embed-js", action="store_true",
                    help="embed plotly.js in the HTML (requires `pip install plotly`); default loads it from a CDN")
    args = ap.parse_args()

    df, n_zero_pre = load(args.csv)
    cov, totals, inc_ct, regions = coverage(df)
    perf = performance(df, regions)

    specs = dict(status=fig_status_bar(totals), incomplete=fig_incomplete(inc_ct),
                 explorer=fig_explorer(perf, regions), heatmap=fig_heatmap(perf, regions),
                 hake=fig_hake(perf, regions), coverage=fig_coverage(cov, totals),
                 composition=fig_composition(cov), table=fig_table(cov, totals))

    plotly_tag = f'<script src="{PLOTLYJS_CDN}" charset="utf-8"></script>'
    if args.embed_js:
        try:
            from plotly.offline import get_plotlyjs
            plotly_tag = f'<script charset="utf-8">{get_plotlyjs()}</script>'
        except ImportError:
            print("plotly not installed; falling back to the CDN script tag.", file=sys.stderr)

    html = build_html(specs, totals, inc_ct, n_zero_pre, perf, args.asof, plotly_tag)
    privacy_check(html, df)
    Path(args.output).write_text(html, encoding="utf-8")

    g = perf["All regions"]["paired"]
    print(f"Wrote {args.output} ({len(html) / 1024:.0f} KB)")
    print(f"Main sites {totals['main_total']}: main trained {totals['main_trained']}, via alternate "
          f"{totals['alt_covered']}, neither {totals['neither']} -> coverage {totals['pct']:.1f}%")
    print(f"Incomplete: {totals['incomplete']} | paired n={g['n']} g={g['g']:.2f} p={g['p']}")


if __name__ == "__main__":
    main()
