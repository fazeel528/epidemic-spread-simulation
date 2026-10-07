"""
utils.py
--------
Visualization and analytics utilities for the Epidemic Spread Simulator.
Uses Plotly for interactive, production-quality charts.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from sir_model import SIRResult


# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "susceptible": "#3B82F6",   # blue
    "infected":    "#EF4444",   # red
    "recovered":   "#10B981",   # green
    "vaccinated":  "#8B5CF6",   # purple
    "no_vax":      "#EF4444",   # red  (comparison)
    "with_vax":    "#8B5CF6",   # purple (comparison)
    "background":  "#0F172A",   # dark navy
    "grid":        "#1E293B",
    "text":        "#F1F5F9",
    "subtext":     "#94A3B8",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color=COLORS["text"], size=12),
    legend=dict(
        bgcolor="rgba(15,23,42,0.8)",
        bordercolor=COLORS["grid"],
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor=COLORS["grid"],
        showline=True,
        linecolor=COLORS["grid"],
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor=COLORS["grid"],
        showline=True,
        linecolor=COLORS["grid"],
        zeroline=False,
    ),
    margin=dict(l=60, r=30, t=60, b=60),
    hovermode="x unified",
)


def plot_sir_curves(result: SIRResult, title: str = "SIR Model — Epidemic Progression") -> go.Figure:
    """
    Plot S, I, R curves over time.

    Parameters
    ----------
    result : SIRResult from a simulation run
    title  : chart title

    Returns
    -------
    Plotly Figure
    """
    fig = go.Figure()

    hover_fmt = "%{y:,.0f}"

    fig.add_trace(go.Scatter(
        x=result.t, y=result.S,
        name="Susceptible",
        mode="lines",
        line=dict(color=COLORS["susceptible"], width=2.5),
        hovertemplate=f"Susceptible: {hover_fmt}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
    ))

    fig.add_trace(go.Scatter(
        x=result.t, y=result.I,
        name="Infected",
        mode="lines",
        line=dict(color=COLORS["infected"], width=2.5),
        hovertemplate=f"Infected: {hover_fmt}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.08)",
    ))

    fig.add_trace(go.Scatter(
        x=result.t, y=result.R,
        name="Recovered",
        mode="lines",
        line=dict(color=COLORS["recovered"], width=2.5),
        hovertemplate=f"Recovered: {hover_fmt}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.08)",
    ))

    # Peak infected annotation
    peak_day = result.peak_day
    peak_t = result.t[peak_day]
    peak_val = result.peak_infected

    fig.add_vline(
        x=peak_t,
        line_dash="dash",
        line_color=COLORS["infected"],
        line_width=1,
        opacity=0.5,
    )
    fig.add_annotation(
        x=peak_t,
        y=peak_val,
        text=f"  Peak: {peak_val:,.0f}<br>  Day {int(peak_t)}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["infected"],
        arrowwidth=1.5,
        font=dict(color=COLORS["infected"], size=11),
        bgcolor="rgba(15,23,42,0.8)",
        bordercolor=COLORS["infected"],
        borderwidth=1,
    )

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"]), x=0.02),
        xaxis_title="Days",
        yaxis_title="Population",
        height=480,
    )
    fig.update_layout(**layout)

    return fig


def plot_comparison(result_no_vax: SIRResult, result_with_vax: SIRResult) -> go.Figure:
    """
    Plot infected curves for two scenarios: no vaccination vs with vaccination.

    Parameters
    ----------
    result_no_vax   : SIRResult without vaccination
    result_with_vax : SIRResult with vaccination

    Returns
    -------
    Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result_no_vax.t, y=result_no_vax.I,
        name="Infected — No Vaccination",
        mode="lines",
        line=dict(color=COLORS["no_vax"], width=2.5),
        hovertemplate="No Vaccination: %{y:,.0f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.10)",
    ))

    fig.add_trace(go.Scatter(
        x=result_with_vax.t, y=result_with_vax.I,
        name=f"Infected — {result_with_vax.vaccination_pct:.0f}% Vaccinated",
        mode="lines",
        line=dict(color=COLORS["with_vax"], width=2.5),
        hovertemplate=f"{result_with_vax.vaccination_pct:.0f}% Vaccinated: "
                      "%{y:,.0f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.10)",
    ))

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(
        title=dict(
            text="Vaccination Impact — Infected Population Comparison",
            font=dict(size=16, color=COLORS["text"]),
            x=0.02,
        ),
        xaxis_title="Days",
        yaxis_title="Infected Population",
        height=420,
    )
    fig.update_layout(**layout)

    return fig


def compute_analytics(result: SIRResult) -> dict:
    """
    Compute key epidemic analytics from a simulation result.

    Returns
    -------
    dict with: peak_infected, peak_day, final_recovered, attack_rate,
               final_susceptible, days_above_threshold
    """
    attack_rate = (result.final_recovered / result.N) * 100  # % of population infected overall

    # Days where infected > 1% of population
    threshold = 0.01 * result.N
    days_above = int(np.sum(result.I > threshold) * (result.t[-1] / len(result.t)))

    return {
        "peak_infected": result.peak_infected,
        "peak_day": int(result.t[result.peak_day]),
        "final_recovered": result.final_recovered,
        "final_susceptible": result.final_susceptible,
        "attack_rate": attack_rate,
        "days_above_threshold": days_above,
    }


def r0_interpretation(r0: float) -> tuple[str, str]:
    """
    Return (status label, explanation) for a given R₀ value.

    Parameters
    ----------
    r0 : basic reproduction number (beta / gamma)

    Returns
    -------
    (status, description) tuple
    """
    if r0 > 2.5:
        return (
            "🔴 Highly Epidemic",
            f"R₀ = {r0:.2f} — Each infected person infects {r0:.1f} others on average. "
            "Rapid, widespread outbreak expected.",
        )
    elif r0 > 1.0:
        return (
            "🟠 Epidemic",
            f"R₀ = {r0:.2f} — Disease will spread through the population. "
            "Intervention required to contain it.",
        )
    elif r0 == 1.0:
        return (
            "🟡 Endemic Threshold",
            f"R₀ = {r0:.2f} — Disease is at the tipping point. "
            "Small changes in beta or gamma will determine outcome.",
        )
    else:
        return (
            "🟢 Contained",
            f"R₀ = {r0:.2f} — Disease will die out naturally. "
            "Each infected person infects fewer than 1 other on average.",
        )


def herd_immunity_threshold(r0: float) -> Optional[float]:
    """
    Compute herd immunity threshold: HIT = 1 - 1/R₀.
    Only meaningful when R₀ > 1.

    Parameters
    ----------
    r0 : basic reproduction number

    Returns
    -------
    HIT as a percentage, or None if R₀ ≤ 1
    """
    if r0 <= 1:
        return None
    return (1.0 - 1.0 / r0) * 100.0
