"""
Generates a self-contained Power BI-style HTML dashboard from scanner results.
All charts are Plotly interactive figures embedded in a single .html file.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── palette ───────────────────────────────────────────────────────────────────
BG_PAGE   = "#0b0c1e"
BG_CARD   = "#13142a"
BG_CHART  = "#181930"
BORDER    = "#2a2b4a"
TEXT_PRI  = "#e8eaf6"
TEXT_SEC  = "#8b8fa8"
CYAN      = "#00d4ff"
ORANGE    = "#ff6b35"
GREEN     = "#10b981"
PURPLE    = "#7c3aed"
RED       = "#ef4444"
YELLOW    = "#f59e0b"

_CHART_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_CHART,
    font=dict(color=TEXT_PRI, family="Inter, Segoe UI, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor=BG_CARD, bordercolor=BORDER, font_color=TEXT_PRI),
)

def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_PRI), x=0.01),
        **_CHART_DEFAULTS,
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont_color=TEXT_SEC)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont_color=TEXT_SEC)
    return fig


# ── individual charts ────────────────────────────────────────────────────────

def chart_top_symbols(mispricings: pd.DataFrame) -> go.Figure:
    by_sym = (
        mispricings.groupby("symbol")
        .agg(count=("abs_mismatch_hist", "size"), avg_abs=("abs_mismatch_hist", "mean"))
        .sort_values("count", ascending=False)
        .head(15)
        .reset_index()
    )
    fig = go.Figure(go.Bar(
        x=by_sym["count"],
        y=by_sym["symbol"],
        orientation="h",
        marker=dict(
            color=by_sym["avg_abs"],
            colorscale=[[0, PURPLE], [0.5, CYAN], [1, ORANGE]],
            colorbar=dict(title="Avg ₹ Gap", tickfont_color=TEXT_SEC, title_font_color=TEXT_SEC),
            showscale=True,
        ),
        text=by_sym["count"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        textfont_color=TEXT_PRI,
        hovertemplate="<b>%{y}</b><br>Mispricings: %{x:,}<br>Avg gap: ₹%{marker.color:.2f}<extra></extra>",
    ))
    fig.update_yaxes(categoryorder="total ascending")
    return _apply_defaults(fig, "Top 15 Symbols — Mismatch Count")


def chart_signal_donut(df: pd.DataFrame, mispricings: pd.DataFrame) -> go.Figure:
    underpriced = int((mispricings["edge_hist"] > 0).sum())
    overpriced  = int((mispricings["edge_hist"] < 0).sum())
    neutral     = int(len(df) - len(mispricings))
    fig = go.Figure(go.Pie(
        labels=["Underpriced (Buy)", "Overpriced (Sell)", "Within Threshold"],
        values=[underpriced, overpriced, neutral],
        hole=0.62,
        marker=dict(colors=[GREEN, RED, "#2d2f4e"], line=dict(color=BG_CARD, width=3)),
        textfont_color=TEXT_PRI,
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"{len(mispricings):,}<br><span style='font-size:10px;color:{TEXT_SEC}'>mispricings</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color=TEXT_PRI),
        align="center",
    )
    return _apply_defaults(fig, "Signal Distribution")


def chart_moneyness_scatter(mispricings: pd.DataFrame) -> go.Figure:
    df = mispricings.copy()
    df["moneyness_pct"] = df["moneyness"] * 100
    df["rel_mismatch_pct"] = df["rel_mismatch_hist"] * 100
    df["abs_clipped"] = df["abs_mismatch_hist"].clip(upper=df["abs_mismatch_hist"].quantile(0.95))

    ce = df[df["option_type"] == "CE"]
    pe = df[df["option_type"] == "PE"]

    fig = go.Figure()
    for subset, name, color in [(ce, "Call (CE)", CYAN), (pe, "Put (PE)", ORANGE)]:
        fig.add_trace(go.Scatter(
            x=subset["moneyness_pct"],
            y=subset["rel_mismatch_pct"],
            mode="markers",
            name=name,
            marker=dict(
                color=color, opacity=0.65, size=subset["abs_clipped"] / subset["abs_clipped"].max() * 14 + 3,
                line=dict(width=0.5, color=BG_CARD),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Moneyness: %{x:.1f}%<br>"
                "Rel mismatch: %{y:.1f}%<br>"
                "Abs gap: ₹%{customdata[1]:.2f}<br>"
                "Market: ₹%{customdata[2]:.2f}<extra></extra>"
            ),
            customdata=np.stack([subset["symbol"], subset["abs_mismatch_hist"], subset["market_price"]], axis=1),
        ))

    fig.add_vline(x=0, line=dict(color=TEXT_SEC, dash="dash", width=1))
    fig.update_xaxes(title_text="Moneyness (%) — negative = OTM", title_font_color=TEXT_SEC)
    fig.update_yaxes(title_text="Relative Mismatch (%)", title_font_color=TEXT_SEC)
    return _apply_defaults(fig, "Moneyness vs Mismatch")


def chart_ce_vs_pe(mispricings: pd.DataFrame) -> go.Figure:
    grouped = (
        mispricings.groupby(["symbol", "option_type"])["abs_mismatch_hist"]
        .mean()
        .reset_index()
    )
    top_syms = (
        mispricings.groupby("symbol").size().sort_values(ascending=False).head(12).index.tolist()
    )
    grouped = grouped[grouped["symbol"].isin(top_syms)]
    ce = grouped[grouped["option_type"] == "CE"]
    pe = grouped[grouped["option_type"] == "PE"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Call (CE)", x=ce["symbol"], y=ce["abs_mismatch_hist"],
        marker_color=CYAN, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Avg CE gap: ₹%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Put (PE)", x=pe["symbol"], y=pe["abs_mismatch_hist"],
        marker_color=ORANGE, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Avg PE gap: ₹%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(barmode="group", legend=dict(bgcolor="rgba(0,0,0,0)", font_color=TEXT_PRI))
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(title_text="Avg Absolute Mismatch (₹)", title_font_color=TEXT_SEC)
    return _apply_defaults(fig, "Avg Mismatch: CE vs PE by Symbol")


def chart_mismatch_histogram(df: pd.DataFrame) -> go.Figure:
    rel = df["rel_mismatch_hist"].dropna()
    rel_pct = rel[rel < rel.quantile(0.98)] * 100  # clip extreme tail

    fig = go.Figure(go.Histogram(
        x=rel_pct,
        nbinsx=60,
        marker=dict(
            color=rel_pct,
            colorscale=[[0, BG_CHART], [0.3, PURPLE], [0.7, CYAN], [1, ORANGE]],
            line=dict(width=0.3, color=BG_CARD),
        ),
        hovertemplate="Range: %{x:.1f}%<br>Count: %{y:,}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Relative Mismatch (%)", title_font_color=TEXT_SEC)
    fig.update_yaxes(title_text="# Options", title_font_color=TEXT_SEC)
    return _apply_defaults(fig, "Mismatch Distribution (all priced options)")


def chart_vol_comparison(df: pd.DataFrame) -> go.Figure:
    top_syms = (
        df.groupby("symbol").size().sort_values(ascending=False).head(12).index.tolist()
    )
    sub = df[df["symbol"].isin(top_syms)]
    avg = sub.groupby("symbol")[["hist_vol", "iv_provided"]].mean().reindex(top_syms).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Hist Vol (30d)",
        x=avg["symbol"],
        y=(avg["hist_vol"] * 100).round(1),
        marker_color=CYAN, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Hist vol: %{y:.1f}%<extra></extra>",
    ))
    valid_iv = avg["iv_provided"].notna()
    fig.add_trace(go.Bar(
        name="Implied Vol (market)",
        x=avg.loc[valid_iv, "symbol"],
        y=(avg.loc[valid_iv, "iv_provided"] * 100).round(1),
        marker_color=ORANGE, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Implied vol: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(barmode="group", legend=dict(bgcolor="rgba(0,0,0,0)", font_color=TEXT_PRI))
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(title_text="Annualised Volatility (%)", title_font_color=TEXT_SEC)
    return _apply_defaults(fig, "Historical Vol vs Implied Vol (top symbols)")


def chart_temporal_breakdown(df: pd.DataFrame) -> go.Figure:
    """
    Shows per-date: symbol count, mismatch rate, and VRP.
    Reveals why April 19-23 looks 'worse': only NIFTY weekly options exist then.
    """
    df2 = df.copy()
    df2["trade_date"] = pd.to_datetime(df2["date"]).dt.normalize()

    daily = (
        df2.groupby("trade_date")
        .agg(
            total       = ("market_price", "size"),
            sym_count   = ("symbol", "nunique"),
            mis_rate    = ("rel_mismatch_hist", lambda x: (x >= 0.15).mean() * 100),
            avg_hist_vol= ("hist_vol", "mean"),
            avg_iv      = ("iv_provided", "mean"),
        )
        .reset_index()
        .sort_values("trade_date")
    )
    daily["vrp"]      = (daily["avg_hist_vol"] - daily["avg_iv"]) * 100
    daily["date_str"] = daily["trade_date"].dt.strftime("%b %d")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Options in dataset per date (symbols count)", "Mismatch rate (%)", "Variance Risk Premium — hist_vol minus IV (%)"],
    )

    # ── row 1: stacked bar of symbol count vs total options ──
    fig.add_trace(go.Bar(
        x=daily["date_str"], y=daily["total"],
        name="Total options",
        marker_color=PURPLE, opacity=0.7,
        hovertemplate="<b>%{x}</b><br>Options: %{y:,}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=daily["date_str"], y=daily["sym_count"],
        name="# Symbols",
        mode="lines+markers+text",
        text=daily["sym_count"].apply(lambda v: f"{v}"),
        textposition="top center",
        textfont=dict(size=9, color=CYAN),
        line=dict(color=CYAN, width=2),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Symbols: %{y}<extra></extra>",
    ), row=1, col=1)

    # ── row 2: mismatch rate ──
    colors_rate = [RED if v > 90 else ORANGE if v > 60 else YELLOW for v in daily["mis_rate"]]
    fig.add_trace(go.Bar(
        x=daily["date_str"], y=daily["mis_rate"].round(1),
        name="Mismatch rate",
        marker_color=colors_rate,
        text=daily["mis_rate"].apply(lambda v: f"{v:.0f}%"),
        textposition="outside",
        textfont_color=TEXT_PRI,
        hovertemplate="<b>%{x}</b><br>Mismatch rate: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=50, line=dict(color=TEXT_SEC, dash="dot", width=1), row=2, col=1)

    # ── row 3: VRP ──
    vrp_colors = [GREEN if v >= 0 else RED for v in daily["vrp"]]
    fig.add_trace(go.Bar(
        x=daily["date_str"], y=daily["vrp"].round(2),
        name="VRP",
        marker_color=vrp_colors,
        text=daily["vrp"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
        textfont_color=TEXT_PRI,
        hovertemplate="<b>%{x}</b><br>VRP: %{y:+.2f}%<br>Hist vol: %{customdata[0]:.1f}%<br>IV: %{customdata[1]:.1f}%<extra></extra>",
        customdata=np.stack([daily["avg_hist_vol"]*100, daily["avg_iv"]*100], axis=1),
        showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0, line=dict(color=TEXT_SEC, dash="dot", width=1), row=3, col=1)

    # region annotation: NIFTY-only zone
    nifty_only_dates = daily[daily["sym_count"] == 1]["date_str"].tolist()
    if nifty_only_dates:
        fig.add_vrect(
            x0=nifty_only_dates[0], x1=nifty_only_dates[-1],
            fillcolor=ORANGE, opacity=0.07,
            layer="below", line_width=0,
            annotation_text="NIFTY<br>weekly<br>only",
            annotation_position="top left",
            annotation_font=dict(size=9, color=ORANGE),
            row="all", col=1,
        )

    fig.update_layout(
        height=640,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color=TEXT_PRI, orientation="h", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_CHART,
        font=dict(color=TEXT_PRI, family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=50, r=20, t=60, b=40),
        hoverlabel=dict(bgcolor=BG_CARD, bordercolor=BORDER, font_color=TEXT_PRI),
        title=dict(text="Temporal Breakdown — Why April 19-23 Looks Different", font=dict(size=14, color=TEXT_PRI), x=0.01),
        yaxis2=dict(
            overlaying="y", side="right",
            showgrid=False, tickfont_color=CYAN, title="# Symbols",
        ),
    )
    fig.update_xaxes(gridcolor=BORDER, tickfont_color=TEXT_SEC)
    fig.update_yaxes(gridcolor=BORDER, tickfont_color=TEXT_SEC)
    return fig


def chart_timeline(mispricings: pd.DataFrame) -> go.Figure:
    daily = (
        mispricings.copy()
        .assign(trade_date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
        .groupby("trade_date")
        .agg(count=("abs_mismatch_hist", "size"), avg_gap=("abs_mismatch_hist", "mean"))
        .reset_index()
        .sort_values("trade_date")
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=daily["trade_date"], y=daily["count"],
        name="Daily count", marker_color=PURPLE, opacity=0.7,
        hovertemplate="<b>%{x|%b %d}</b><br>Count: %{y:,}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=daily["trade_date"], y=daily["avg_gap"],
        name="Avg ₹ gap", mode="lines+markers",
        line=dict(color=CYAN, width=2), marker=dict(size=4),
        hovertemplate="<b>%{x|%b %d}</b><br>Avg gap: ₹%{y:.2f}<extra></extra>",
    ), secondary_y=True)
    fig.update_yaxes(title_text="Mismatch count", title_font_color=TEXT_SEC, secondary_y=False,
                     gridcolor=BORDER, tickfont_color=TEXT_SEC)
    fig.update_yaxes(title_text="Avg absolute gap (₹)", title_font_color=TEXT_SEC, secondary_y=True,
                     gridcolor="rgba(0,0,0,0)", tickfont_color=TEXT_SEC)
    fig.update_layout(
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color=TEXT_PRI),
        **_CHART_DEFAULTS,
        title=dict(text="Daily Mismatch Trend", font=dict(size=14, color=TEXT_PRI), x=0.01),
    )
    fig.update_xaxes(gridcolor=BORDER, tickfont_color=TEXT_SEC)
    return fig


# ── table HTML ────────────────────────────────────────────────────────────────

def _table_html(mispricings: pd.DataFrame, top_n: int = 100) -> str:
    cols = [
        "symbol", "date", "option_type", "strike", "expiry",
        "market_price", "bs_price_hist", "abs_mismatch_hist",
        "rel_mismatch_hist", "edge_hist", "moneyness",
        "spot", "hist_vol", "T", "volume", "oi",
    ]
    df = (
        mispricings[[c for c in cols if c in mispricings.columns]]
        .sort_values("abs_mismatch_hist", ascending=False)
        .head(top_n)
        .copy()
    )

    fmt = {
        "date":              lambda v: pd.Timestamp(v).strftime("%Y-%m-%d"),
        "expiry":            lambda v: pd.Timestamp(v).strftime("%Y-%m-%d"),
        "market_price":      lambda v: f"₹{v:.2f}",
        "bs_price_hist":     lambda v: f"₹{v:.2f}",
        "abs_mismatch_hist": lambda v: f"₹{v:.2f}",
        "rel_mismatch_hist": lambda v: f"{v*100:.1f}%",
        "edge_hist":         lambda v: f"₹{v:.2f}",
        "moneyness":         lambda v: f"{v*100:.1f}%",
        "hist_vol":          lambda v: f"{v*100:.1f}%",
        "T":                 lambda v: f"{v*365:.0f}d",
        "volume":            lambda v: f"{int(v):,}",
        "oi":                lambda v: f"{int(v):,}",
        "strike":            lambda v: f"{int(v):,}",
        "spot":              lambda v: f"₹{v:.1f}",
    }

    header_names = {
        "symbol": "Symbol", "date": "Date", "option_type": "Type",
        "strike": "Strike", "expiry": "Expiry",
        "market_price": "Market", "bs_price_hist": "BS Price",
        "abs_mismatch_hist": "Gap ₹", "rel_mismatch_hist": "Gap %",
        "edge_hist": "Edge", "moneyness": "Moneyness",
        "spot": "Spot", "hist_vol": "Hist Vol", "T": "DTE",
        "volume": "Volume", "oi": "OI",
    }

    thead = "".join(
        f'<th onclick="sortTable({i})" title="click to sort">'
        f'{header_names.get(c, c)} <span class="sort-icon">⇅</span></th>'
        for i, c in enumerate(df.columns)
    )

    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            val  = row[c]
            text = fmt[c](val) if c in fmt else str(val)

            cls = ""
            if c == "edge_hist":
                cls = "positive" if val > 0 else "negative"
            elif c == "option_type":
                cls = "ce" if val == "CE" else "pe"
            elif c == "rel_mismatch_hist":
                cls = "high-miss" if val > 0.5 else ""

            cells.append(f'<td class="{cls}">{text}</td>')
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
<div class="table-wrapper">
  <table id="mismatch-table">
    <thead><tr>{thead}</tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</div>
"""


# ── diagnostic / root-cause charts ───────────────────────────────────────────

def chart_greeks_availability(df: pd.DataFrame, mispricings: pd.DataFrame) -> go.Figure:
    """Bar showing how mismatch collapses when IV is used instead of hist_vol."""
    has_iv = df["iv_provided"].notna()

    grp_has = df[has_iv]
    grp_not = df[~has_iv]

    iv_rel_has = (grp_has["abs_mismatch_iv"] / grp_has["market_price"]).mean() * 100

    categories = ["Has IV (Greeks)", "No IV (no Greeks)"]
    hist_vals  = [grp_has["rel_mismatch_hist"].mean() * 100,
                  grp_not["rel_mismatch_hist"].mean() * 100]
    iv_vals    = [iv_rel_has, None]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Using hist_vol",
        x=categories, y=hist_vals,
        marker_color=ORANGE, opacity=0.85,
        text=[f"{v:.1f}%" for v in hist_vals],
        textposition="outside", textfont_color=TEXT_PRI,
        hovertemplate="<b>%{x}</b><br>Hist-vol mismatch: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Using IV (market's own vol)",
        x=[categories[0]], y=[iv_rel_has],
        marker_color=GREEN, opacity=0.85,
        text=[f"{iv_rel_has:.1f}%"],
        textposition="outside", textfont_color=TEXT_PRI,
        hovertemplate="<b>%{x}</b><br>IV mismatch: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(barmode="group", legend=dict(bgcolor="rgba(0,0,0,0)", font_color=TEXT_PRI))
    fig.update_yaxes(title_text="Avg Relative Mismatch (%)", title_font_color=TEXT_SEC)
    return _apply_defaults(fig, "Mismatch: hist_vol vs IV — missing Greeks NOT the cause")


def chart_variance_risk_premium(df: pd.DataFrame) -> go.Figure:
    """Hist_vol − IV (VRP) per symbol — the primary driver of 'mispricing'."""
    has_iv = df["iv_provided"].notna()
    grp = df[has_iv].copy()

    sym_vrp = (
        grp.groupby("symbol")
        .apply(lambda g: pd.Series({
            "vrp":      (g["hist_vol"] - g["iv_provided"]).mean() * 100,
            "hist_vol": g["hist_vol"].mean() * 100,
            "iv":       g["iv_provided"].mean() * 100,
            "n":        len(g),
        }))
        .reset_index()
        .sort_values("vrp", ascending=True)
    )
    # top symbols by count
    top = df.groupby("symbol").size().sort_values(ascending=False).head(20).index
    sym_vrp = sym_vrp[sym_vrp["symbol"].isin(top)].sort_values("vrp", ascending=True)

    colors = [GREEN if v >= 0 else RED for v in sym_vrp["vrp"]]

    fig = go.Figure(go.Bar(
        x=sym_vrp["vrp"],
        y=sym_vrp["symbol"],
        orientation="h",
        marker_color=colors,
        opacity=0.85,
        text=sym_vrp["vrp"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
        textfont_color=TEXT_PRI,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "VRP: %{x:+.2f}%<br>"
            "Hist vol: %{customdata[0]:.1f}%<br>"
            "Implied vol: %{customdata[1]:.1f}%<extra></extra>"
        ),
        customdata=np.stack([sym_vrp["hist_vol"], sym_vrp["iv"]], axis=1),
    ))
    fig.add_vline(x=0, line=dict(color=TEXT_SEC, dash="dash", width=1))
    fig.update_xaxes(title_text="Hist Vol − Implied Vol (%)", title_font_color=TEXT_SEC)
    fig.update_yaxes(categoryorder="array", categoryarray=sym_vrp["symbol"].tolist())
    return _apply_defaults(fig, "Variance Risk Premium (hist_vol − IV) by Symbol")


def chart_iv_bs_residual(df: pd.DataFrame) -> go.Figure:
    """Distribution of BS(IV) − market: shows systematic dividend-driven upward bias."""
    has_iv = df["iv_provided"].notna()
    grp = df[has_iv].copy()
    residual = grp["bs_price_iv"] - grp["market_price"]
    residual_clipped = residual.clip(
        lower=residual.quantile(0.01),
        upper=residual.quantile(0.99),
    )

    pct_above = (residual > 0).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residual_clipped,
        nbinsx=80,
        marker=dict(
            color=residual_clipped,
            colorscale=[[0, RED], [0.5, PURPLE], [1, GREEN]],
            line=dict(width=0.3, color=BG_CARD),
        ),
        hovertemplate="BS(IV)−Market: %{x:.2f}<br>Count: %{y:,}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=YELLOW, dash="dash", width=1.5),
                  annotation_text="Zero = perfect fit",
                  annotation_font_color=YELLOW)
    fig.add_vline(x=residual.mean(), line=dict(color=ORANGE, width=1.5),
                  annotation_text=f"Mean = ₹{residual.mean():.2f}",
                  annotation_font_color=ORANGE)
    fig.update_xaxes(title_text="BS(IV) − Market Price (₹)", title_font_color=TEXT_SEC)
    fig.update_yaxes(title_text="Count", title_font_color=TEXT_SEC)
    return _apply_defaults(
        fig,
        f"BS(IV) Residual — {pct_above:.1f}% above market (dividend yield omission)"
    )


def chart_iv_vs_histvol_scatter(df: pd.DataFrame) -> go.Figure:
    """IV vs hist_vol scatter — shows VRP regime (which stocks run hot vs cool)."""
    has_iv = df["iv_provided"].notna()
    grp = df[has_iv].copy()

    sym_avg = (
        grp.groupby("symbol")[["iv_provided", "hist_vol", "rel_mismatch_hist"]]
        .mean()
        .reset_index()
    )
    sym_avg["iv_pct"]   = sym_avg["iv_provided"] * 100
    sym_avg["hv_pct"]   = sym_avg["hist_vol"] * 100
    sym_avg["vrp"]      = sym_avg["hv_pct"] - sym_avg["iv_pct"]

    fig = go.Figure()
    # 45-degree line = IV == hist_vol (zero VRP)
    ax_max = max(sym_avg["hv_pct"].max(), sym_avg["iv_pct"].max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[0, ax_max], y=[0, ax_max],
        mode="lines",
        line=dict(color=TEXT_SEC, dash="dot", width=1),
        name="IV = Hist Vol (zero VRP)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=sym_avg["hv_pct"],
        y=sym_avg["iv_pct"],
        mode="markers+text",
        text=sym_avg["symbol"],
        textposition="top center",
        textfont=dict(size=9, color=TEXT_SEC),
        marker=dict(
            color=sym_avg["vrp"],
            colorscale=[[0, RED], [0.4, PURPLE], [0.6, PURPLE], [1, GREEN]],
            size=10,
            colorbar=dict(
                title="VRP %", tickfont_color=TEXT_SEC,
                title_font_color=TEXT_SEC,
            ),
            line=dict(width=0.5, color=BG_CARD),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Hist Vol: %{x:.1f}%<br>"
            "Impl Vol: %{y:.1f}%<br>"
            "VRP: %{marker.color:+.1f}%<extra></extra>"
        ),
        name="Symbols",
    ))
    # Annotations
    above_line = sym_avg[sym_avg["vrp"] > 0]
    below_line = sym_avg[sym_avg["vrp"] < 0]
    fig.add_annotation(
        x=ax_max * 0.05, y=ax_max * 0.92,
        text=f"← ABOVE line: hist_vol > IV<br>  Options 'cheap' vs realized vol<br>  ({len(above_line)} symbols)",
        showarrow=False, font=dict(size=10, color=GREEN),
        align="left", bgcolor="rgba(19,20,42,0.85)",
    )
    fig.add_annotation(
        x=ax_max * 0.55, y=ax_max * 0.05,
        text=f"BELOW line: IV > hist_vol →<br>Options 'expensive' vs realized vol<br>({len(below_line)} symbols)",
        showarrow=False, font=dict(size=10, color=RED),
        align="left", bgcolor="rgba(19,20,42,0.85)",
    )
    fig.update_xaxes(title_text="Historical Vol, 30d (%)", title_font_color=TEXT_SEC)
    fig.update_yaxes(title_text="Implied Vol — market IV (%)", title_font_color=TEXT_SEC)
    return _apply_defaults(fig, "IV vs Historical Vol — Variance Risk Premium Map")


def _insight_banner(df: pd.DataFrame, mispricings: pd.DataFrame) -> str:
    has_iv       = df["iv_provided"].notna()
    pct_flags_have_iv = (
        (mispricings["iv_provided"].notna()).sum() / len(mispricings) * 100
        if len(mispricings) else 0
    )
    grp      = df[has_iv]
    avg_vrp  = (grp["hist_vol"] - grp["iv_provided"]).mean() * 100
    residual = (grp["bs_price_iv"] - grp["market_price"])
    pct_above = (residual > 0).mean() * 100

    items = [
        (
            "🔍", RED, "Not Missing Greeks",
            f"<b>{pct_flags_have_iv:.1f}%</b> of all flagged mispricings are in rows "
            f"<em>that DO have</em> exchange-provided IV & Greeks. Missing Greeks "
            f"account for only <b>{100-pct_flags_have_iv:.1f}%</b> of flags.",
        ),
        (
            "📊", ORANGE, "Variance Risk Premium (Primary Driver)",
            f"Historical vol (30d realized) runs <b>+{avg_vrp:.2f}%</b> above implied vol on "
            f"average. Options appear cheap vs realized vol because the market charges a "
            f"<em>volatility risk premium</em> — sellers demand extra for the risk of a vol spike. "
            f"This gap IS the 'mismatch', not a pricing error.",
        ),
        (
            "💰", YELLOW, "Dividend Yield Omission (Systematic Bias)",
            f"Even when plugging in the market's own IV, BS(IV) exceeds market price "
            f"<b>{pct_above:.1f}%</b> of the time (mean bias ₹{residual.mean():.2f}). "
            f"Indian stocks pay dividends that lower the forward price — "
            f"Black-Scholes on spot without subtracting dividend yield systematically "
            f"<em>overstates</em> call prices and understates puts.",
        ),
        (
            "✅", GREEN, "True BS Mismatch Is Small",
            f"Once IV is used instead of hist_vol, the average relative mismatch drops "
            f"from <b>{df['rel_mismatch_hist'].mean()*100:.1f}%</b> → "
            f"<b>{(df['abs_mismatch_iv'].dropna() / df.loc[df['iv_provided'].notna(),'market_price']).mean()*100:.1f}%</b>. "
            f"Remaining residual is explained by dividend adjustments, not market inefficiency.",
        ),
    ]

    cards_html = ""
    for icon, color, title, text in items:
        cards_html += f"""
<div class="insight-card" style="border-left: 3px solid {color}">
  <div class="insight-header">
    <span class="insight-icon">{icon}</span>
    <span class="insight-title" style="color:{color}">{title}</span>
  </div>
  <div class="insight-text">{text}</div>
</div>"""

    return f'<div class="insight-grid">{cards_html}</div>'


# ── KPI cards ─────────────────────────────────────────────────────────────────

def _kpi_cards(df: pd.DataFrame, mispricings: pd.DataFrame,
               gpu_device: str, load_time: float, gpu_time: float) -> str:
    rate = len(mispricings) / len(df) * 100 if len(df) else 0
    avg_gap = mispricings["abs_mismatch_hist"].mean() if not mispricings.empty else 0
    buy  = int((mispricings["edge_hist"] > 0).sum())
    sell = int((mispricings["edge_hist"] < 0).sum())

    cards = [
        ("Total Options", f"{len(df):,}",        "Priced after filters",  CYAN,   "📊"),
        ("Mispricings",   f"{len(mispricings):,}", f"{rate:.1f}% rate",    ORANGE, "⚡"),
        ("Buy Signals",   f"{buy:,}",             "Underpriced vs BS",     GREEN,  "📈"),
        ("Sell Signals",  f"{sell:,}",            "Overpriced vs BS",      RED,    "📉"),
        ("Avg Gap",       f"₹{avg_gap:.2f}",      "Mean abs mismatch",     YELLOW, "💰"),
        ("GPU Time",      f"{gpu_time:.3f}s",     f"{gpu_device.upper()}", PURPLE, "⚙️"),
    ]
    html = '<div class="kpi-row">'
    for title, value, sub, color, icon in cards:
        html += f"""
<div class="kpi-card">
  <div class="kpi-icon">{icon}</div>
  <div class="kpi-body">
    <div class="kpi-value" style="color:{color}">{value}</div>
    <div class="kpi-title">{title}</div>
    <div class="kpi-sub">{sub}</div>
  </div>
</div>"""
    html += "</div>"
    return html


# ── full page assembler ────────────────────────────────────────────────────────

def build_dashboard(
    df:         pd.DataFrame,
    mispricings: pd.DataFrame,
    gpu_device:  str,
    load_time:   float,
    gpu_time:    float,
    out_path:    str | Path = "output/dashboard.html",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _div(fig: go.Figure, h: int = 380) -> str:
        return fig.to_html(
            full_html=False, include_plotlyjs=False,
            config={"displayModeBar": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            default_height=h,
        )

    kpis          = _kpi_cards(df, mispricings, gpu_device, load_time, gpu_time)
    top_sym_div   = _div(chart_top_symbols(mispricings))
    donut_div     = _div(chart_signal_donut(df, mispricings))
    scatter_div   = _div(chart_moneyness_scatter(mispricings))
    ce_pe_div     = _div(chart_ce_vs_pe(mispricings))
    hist_div      = _div(chart_mismatch_histogram(df))
    vol_cmp_div   = _div(chart_vol_comparison(df))
    timeline_div     = _div(chart_timeline(mispricings), h=320)
    temporal_div     = _div(chart_temporal_breakdown(df), h=640)
    table_html    = _table_html(mispricings, top_n=100)

    # diagnostic / root-cause section
    insight_html   = _insight_banner(df, mispricings)
    greeks_div     = _div(chart_greeks_availability(df, mispricings))
    vrp_bar_div    = _div(chart_variance_risk_premium(df))
    iv_residual_div = _div(chart_iv_bs_residual(df))
    iv_scatter_div  = _div(chart_iv_vs_histvol_scatter(df), h=460)

    run_ts = datetime.now().strftime("%d %b %Y, %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BS Mispricing Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: {BG_PAGE};
    color: {TEXT_PRI};
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 13px;
    min-height: 100vh;
  }}

  /* ── header ── */
  .header {{
    background: linear-gradient(135deg, #12132c 0%, #1a1b3c 100%);
    border-bottom: 1px solid {BORDER};
    padding: 18px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  .header-left {{ display: flex; align-items: center; gap: 14px; }}
  .header-logo {{
    width: 38px; height: 38px;
    background: linear-gradient(135deg, {CYAN}, {PURPLE});
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }}
  .header-title {{ font-size: 17px; font-weight: 600; color: {TEXT_PRI}; }}
  .header-sub   {{ font-size: 11px; color: {TEXT_SEC}; margin-top: 2px; }}
  .header-meta  {{ text-align: right; color: {TEXT_SEC}; font-size: 11px; line-height: 1.7; }}
  .header-meta span {{ color: {CYAN}; font-weight: 500; }}

  /* ── main layout ── */
  .main {{ padding: 22px 28px; max-width: 1600px; margin: 0 auto; }}

  /* ── section label ── */
  .section-label {{
    font-size: 11px; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: {TEXT_SEC};
    margin: 24px 0 10px;
    display: flex; align-items: center; gap: 8px;
  }}
  .section-label::after {{
    content: ''; flex: 1; height: 1px; background: {BORDER};
  }}

  /* ── KPI cards ── */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 14px;
    margin-bottom: 4px;
  }}
  @media (max-width: 1200px) {{ .kpi-row {{ grid-template-columns: repeat(3, 1fr); }} }}
  @media (max-width: 700px)  {{ .kpi-row {{ grid-template-columns: repeat(2, 1fr); }} }}

  .kpi-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: border-color 0.2s;
  }}
  .kpi-card:hover {{ border-color: {CYAN}44; }}
  .kpi-icon {{ font-size: 22px; flex-shrink: 0; }}
  .kpi-value {{ font-size: 22px; font-weight: 700; line-height: 1.1; }}
  .kpi-title {{ font-size: 11px; font-weight: 500; color: {TEXT_SEC}; margin-top: 3px; }}
  .kpi-sub   {{ font-size: 10px; color: {TEXT_SEC}44; margin-top: 1px; }}

  /* ── chart cards ── */
  .chart-grid-2 {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
  }}
  .chart-grid-3 {{
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px;
  }}
  @media (max-width: 1100px) {{ .chart-grid-2, .chart-grid-3 {{ grid-template-columns: 1fr; }} }}

  .chart-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
    transition: border-color 0.2s;
  }}
  .chart-card:hover {{ border-color: {CYAN}44; }}

  .chart-card.wide {{
    grid-column: 1 / -1;
  }}

  /* ── table ── */
  .table-section-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px 0;
  }}
  .table-section-title {{
    font-size: 13px; font-weight: 600; color: {TEXT_PRI};
  }}
  .table-section-sub {{
    font-size: 11px; color: {TEXT_SEC};
  }}
  .table-search {{
    background: {BG_CHART};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 12px;
    color: {TEXT_PRI};
    font-size: 12px;
    outline: none;
    width: 200px;
  }}
  .table-search:focus {{ border-color: {CYAN}; }}

  .table-wrapper {{
    overflow-x: auto;
    padding: 12px;
    max-height: 500px;
    overflow-y: auto;
  }}
  .table-wrapper::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  .table-wrapper::-webkit-scrollbar-track {{ background: {BG_CARD}; }}
  .table-wrapper::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

  #mismatch-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    white-space: nowrap;
  }}
  #mismatch-table thead {{
    position: sticky; top: 0; z-index: 2;
    background: {BG_CARD};
  }}
  #mismatch-table th {{
    padding: 9px 12px;
    text-align: left;
    color: {TEXT_SEC};
    font-weight: 500;
    font-size: 11px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    cursor: pointer;
    user-select: none;
    border-bottom: 1px solid {BORDER};
    white-space: nowrap;
  }}
  #mismatch-table th:hover {{ color: {CYAN}; }}
  .sort-icon {{ opacity: 0.4; font-size: 10px; }}
  #mismatch-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid {BORDER}22;
    color: {TEXT_PRI};
  }}
  #mismatch-table tr:hover td {{ background: {BORDER}33; }}
  #mismatch-table .positive {{ color: {GREEN}; font-weight: 500; }}
  #mismatch-table .negative {{ color: {RED};   font-weight: 500; }}
  #mismatch-table .ce {{ color: {CYAN};   font-weight: 600; }}
  #mismatch-table .pe {{ color: {ORANGE}; font-weight: 600; }}
  #mismatch-table .high-miss {{ color: {YELLOW}; font-weight: 600; }}

  /* ── insight cards ── */
  .insight-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
  }}
  @media (max-width: 900px) {{ .insight-grid {{ grid-template-columns: 1fr; }} }}

  .insight-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    transition: border-color 0.2s;
  }}
  .insight-card:hover {{ border-color: {CYAN}44; }}
  .insight-header {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
  }}
  .insight-icon  {{ font-size: 18px; }}
  .insight-title {{ font-size: 13px; font-weight: 600; }}
  .insight-text  {{ font-size: 12px; color: {TEXT_SEC}; line-height: 1.7; }}
  .insight-text b {{ color: {TEXT_PRI}; font-weight: 600; }}
  .insight-text em {{ color: {CYAN}; font-style: normal; }}

  /* ── footer ── */
  .footer {{
    text-align: center;
    padding: 20px;
    color: {TEXT_SEC};
    font-size: 11px;
    border-top: 1px solid {BORDER};
    margin-top: 28px;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="header-logo">📐</div>
    <div>
      <div class="header-title">Black-Scholes Mispricing Scanner</div>
      <div class="header-sub">Options pricing analysis vs historical volatility</div>
    </div>
  </div>
  <div class="header-meta">
    Generated: <span>{run_ts}</span><br>
    Device: <span>{gpu_device.upper()}</span> &nbsp;|&nbsp;
    Load: <span>{load_time:.1f}s</span> &nbsp;|&nbsp;
    GPU Pricing: <span>{gpu_time:.3f}s</span>
  </div>
</div>

<div class="main">

  <div class="section-label">Key Metrics</div>
  {kpis}

  <div class="section-label">Mismatch Overview</div>
  <div class="chart-grid-2">
    <div class="chart-card">{top_sym_div}</div>
    <div class="chart-card">{donut_div}</div>
  </div>

  <div class="section-label">Deep Dive</div>
  <div class="chart-grid-2">
    <div class="chart-card">{scatter_div}</div>
    <div class="chart-card">{ce_pe_div}</div>
  </div>

  <div class="section-label">Volatility & Distribution</div>
  <div class="chart-grid-2">
    <div class="chart-card">{vol_cmp_div}</div>
    <div class="chart-card">{hist_div}</div>
  </div>

  <div class="section-label">Why So Much Mispricing? — Root Cause Analysis</div>
  {insight_html}

  <div class="chart-grid-2" style="margin-top:14px">
    <div class="chart-card">{greeks_div}</div>
    <div class="chart-card">{vrp_bar_div}</div>
  </div>
  <div class="chart-grid-2">
    <div class="chart-card">{iv_residual_div}</div>
    <div class="chart-card">{iv_scatter_div}</div>
  </div>

  <div class="section-label">Why Only April? — Temporal Breakdown</div>
  <div class="chart-card wide">{temporal_div}</div>

  <div class="section-label">Timeline</div>
  <div class="chart-grid-2">
    <div class="chart-card wide">{timeline_div}</div>
  </div>

  <div class="section-label">Top 100 Mispricings</div>
  <div class="chart-card wide">
    <div class="table-section-header">
      <div>
        <div class="table-section-title">Top 100 Mispricings — sorted by absolute gap</div>
        <div class="table-section-sub">
          Green edge = underpriced (buy signal) · Red edge = overpriced (sell signal)
        </div>
      </div>
      <input class="table-search" type="text" id="tableSearch" placeholder="🔍  Filter by symbol…" oninput="filterTable(this.value)">
    </div>
    {table_html}
  </div>

</div>

<div class="footer">
  Black-Scholes Mispricing Scanner &nbsp;·&nbsp; Risk-free rate 7% &nbsp;·&nbsp; 30-day rolling hist vol &nbsp;·&nbsp;
  Mismatch = |BS(σ_hist) − market| / market
</div>

<script>
// ── table sort ──────────────────────────────────────────────────────────────
let sortDir = {{}};
function sortTable(col) {{
  const tbl  = document.getElementById("mismatch-table");
  const rows = Array.from(tbl.tBodies[0].rows);
  sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    let av = a.cells[col].innerText.replace(/[₹,%,]/g, "").trim();
    let bv = b.cells[col].innerText.replace(/[₹,%,]/g, "").trim();
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return sortDir[col] ? an - bn : bn - an;
    return sortDir[col] ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbl.tBodies[0].appendChild(r));
  // update sort icons
  tbl.querySelectorAll(".sort-icon").forEach((el, i) => {{
    el.textContent = i === col ? (sortDir[col] ? "▲" : "▼") : "⇅";
    el.style.opacity = i === col ? "1" : "0.4";
  }});
}}

// ── table filter ─────────────────────────────────────────────────────────────
function filterTable(query) {{
  const q    = query.toLowerCase();
  const rows = document.querySelectorAll("#mismatch-table tbody tr");
  rows.forEach(r => {{
    r.style.display = r.cells[0].innerText.toLowerCase().includes(q) ? "" : "none";
  }});
}}
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    return out_path
