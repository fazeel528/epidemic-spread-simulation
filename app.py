"""
app.py
------
Streamlit UI for the Epidemic Spread Simulator (SIR Model).
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np

from sir_model import SIRParameters, run_simulation
from utils import (
    plot_sir_curves,
    plot_comparison,
    compute_analytics,
    r0_interpretation,
    herd_immunity_threshold,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Epidemic Spread Simulator",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0F172A;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] .stSlider > label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
    color: #94A3B8 !important;
}

/* Title block */
.title-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #F1F5F9;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}
.subtitle-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #64748B;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
    margin-bottom: 24px;
}

/* Metric cards */
.metric-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #64748B;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #F1F5F9;
    line-height: 1.1;
}
.metric-value.blue  { color: #3B82F6; }
.metric-value.red   { color: #EF4444; }
.metric-value.green { color: #10B981; }
.metric-value.amber { color: #F59E0B; }

/* Info / status boxes */
.status-box {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    background: rgba(15, 23, 42, 0.8);
    border-left: 3px solid;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 12px 0;
    color: #CBD5E1;
    line-height: 1.6;
}
.status-box.epidemic  { border-color: #EF4444; }
.status-box.contained { border-color: #10B981; }
.status-box.warning   { border-color: #F59E0B; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
}

/* Divider */
hr { border-color: #1E293B; }

/* Checkbox */
.stCheckbox > label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #94A3B8 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar — Input Parameters ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦠 Parameters")
    st.markdown("---")

    st.markdown("**Population**")
    N = st.slider("Total Population (N)", 1_000, 10_000_000, 1_000_000, step=1_000,
                  format="%d")

    I0 = st.slider("Initial Infected (I₀)", 1, max(1, N // 100), 10, step=1)

    max_r0 = max(0, N - I0)
    R0_initial = st.slider("Initial Recovered (R₀ initial)", 0, max_r0,
                            min(0, max_r0), step=1)

    st.markdown("---")
    st.markdown("**Disease Parameters**")
    beta = st.slider("Infection Rate (β)", 0.01, 1.0, 0.30, step=0.01,
                     help="Average contacts per day × probability of transmission")
    gamma = st.slider("Recovery Rate (γ)", 0.01, 1.0, 0.10, step=0.01,
                      help="1/γ = average infectious period in days")
    days = st.slider("Simulation Duration (days)", 30, 1000, 160, step=10)

    st.markdown("---")
    st.markdown("**Vaccination**")
    vaccination_pct = st.slider("Vaccination Coverage (%)", 0.0, 100.0, 0.0, step=0.5,
                                 help="% of population vaccinated before simulation begins")

    st.markdown("---")
    st.markdown("**Comparison Mode**")
    compare_enabled = st.checkbox("Enable Vaccination Comparison", value=False,
                                   help="Plot infected curves with and without vaccination side by side")

    st.markdown("---")
    # Quick reference
    st.markdown(
        "<span style='font-size:0.7rem;color:#475569;font-family:IBM Plex Mono'>β/γ = R₀  "
        "| 1/γ = infectious days</span>",
        unsafe_allow_html=True,
    )


# ── Validate & Build Parameters ───────────────────────────────────────────────
try:
    params = SIRParameters(
        N=N,
        I0=I0,
        R0_initial=R0_initial,
        beta=beta,
        gamma=gamma,
        days=days,
        vaccination_pct=vaccination_pct,
    )
    param_error = None
except ValueError as e:
    param_error = str(e)
    params = None


# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown('<div class="title-block">🦠 Epidemic Spread Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-block">SIR Model · Numerical Integration · Real-time Analysis</div>',
            unsafe_allow_html=True)

if param_error:
    st.error(f"⚠️ Invalid parameters: {param_error}")
    st.stop()


# ── Run Simulations ────────────────────────────────────────────────────────────
result = run_simulation(params)

if compare_enabled and vaccination_pct > 0:
    result_no_vax = run_simulation(params, vaccination_pct=0.0)
    result_with_vax = run_simulation(params, vaccination_pct=vaccination_pct)
elif compare_enabled and vaccination_pct == 0:
    st.warning("Set vaccination coverage > 0% to enable comparison mode.", icon="⚠️")
    compare_enabled = False


# ── R₀ Status Banner ──────────────────────────────────────────────────────────
r0_value = params.basic_reproduction_number
r0_label, r0_desc = r0_interpretation(r0_value)
hit = herd_immunity_threshold(r0_value)

box_class = "epidemic" if r0_value > 1 else "contained"
st.markdown(
    f'<div class="status-box {box_class}">'
    f'<strong>{r0_label}</strong><br>{r0_desc}'
    + (f'<br><br>💉 Herd immunity threshold: <strong>{hit:.1f}%</strong> vaccination needed'
       if hit else "")
    + "</div>",
    unsafe_allow_html=True,
)


# ── Analytics Metrics ──────────────────────────────────────────────────────────
analytics = compute_analytics(result)
infectious_period = 1 / gamma

st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Peak Infected</div>'
        f'<div class="metric-value red">{analytics["peak_infected"]:,.0f}</div></div>',
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Day of Peak</div>'
        f'<div class="metric-value amber">Day {analytics["peak_day"]}</div></div>',
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Total Recovered</div>'
        f'<div class="metric-value green">{analytics["final_recovered"]:,.0f}</div></div>',
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Attack Rate</div>'
        f'<div class="metric-value blue">{analytics["attack_rate"]:.1f}%</div></div>',
        unsafe_allow_html=True,
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Basic R₀</div>'
        f'<div class="metric-value {"red" if r0_value > 1 else "green"}">{r0_value:.2f}</div></div>',
        unsafe_allow_html=True,
    )

with col6:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Infectious Period</div>'
        f'<div class="metric-value blue">{infectious_period:.1f} days</div></div>',
        unsafe_allow_html=True,
    )

with col7:
    vaccinated_count = (vaccination_pct / 100) * N
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Vaccinated</div>'
        f'<div class="metric-value blue">{vaccinated_count:,.0f}</div></div>',
        unsafe_allow_html=True,
    )

with col8:
    remaining = analytics["final_susceptible"]
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Never Infected</div>'
        f'<div class="metric-value blue">{remaining:,.0f}</div></div>',
        unsafe_allow_html=True,
    )


# ── Main SIR Chart ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Epidemic Curves</div>', unsafe_allow_html=True)

vax_label = f" | {vaccination_pct:.0f}% Vaccinated" if vaccination_pct > 0 else ""
fig_main = plot_sir_curves(
    result,
    title=f"SIR Model — N={N:,} | β={beta} | γ={gamma}{vax_label}",
)
st.plotly_chart(fig_main, use_container_width=True)


# ── Conservation Check ─────────────────────────────────────────────────────────
if result.verify_conservation():
    st.markdown(
        '<div class="status-box contained" style="font-size:0.75rem;padding:8px 12px;">'
        '✓ Conservation verified — S + I + R = N holds across all time steps.'
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.warning("⚠️ Conservation check failed — numerical drift detected.")


# ── Comparison Chart ───────────────────────────────────────────────────────────
if compare_enabled:
    st.markdown('<div class="section-header">Vaccination Comparison</div>', unsafe_allow_html=True)

    fig_cmp = plot_comparison(result_no_vax, result_with_vax)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Comparison summary
    delta_peak = result_no_vax.peak_infected - result_with_vax.peak_infected
    delta_pct = (delta_peak / result_no_vax.peak_infected * 100) if result_no_vax.peak_infected > 0 else 0
    delta_rec = result_no_vax.final_recovered - result_with_vax.final_recovered

    st.markdown('<div class="section-header">Vaccination Impact</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Peak Reduction</div>'
            f'<div class="metric-value green">{delta_peak:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Peak Reduction %</div>'
            f'<div class="metric-value green">{delta_pct:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Cases Prevented</div>'
            f'<div class="metric-value green">{delta_rec:,.0f}</div></div>',
            unsafe_allow_html=True,
        )


# ── Model Reference ────────────────────────────────────────────────────────────
with st.expander("📐 Model Reference & Equations", expanded=False):
    st.markdown("""
<div style='font-family: IBM Plex Mono, monospace; font-size: 0.82rem; color: #CBD5E1; line-height: 2;'>

**Differential Equations**

&nbsp;&nbsp;&nbsp; dS/dt = −β · S · I / N

&nbsp;&nbsp;&nbsp; dI/dt = &nbsp;β · S · I / N − γ · I

&nbsp;&nbsp;&nbsp; dR/dt = &nbsp;γ · I

**Conservation Law**

&nbsp;&nbsp;&nbsp; S(t) + I(t) + R(t) = N &nbsp; for all t

**Basic Reproduction Number**

&nbsp;&nbsp;&nbsp; R₀ = β / γ

**Herd Immunity Threshold**

&nbsp;&nbsp;&nbsp; HIT = 1 − 1/R₀ &nbsp; (only when R₀ > 1)

**Infectious Period**

&nbsp;&nbsp;&nbsp; D = 1/γ (days)

**Numerical Method:** scipy.integrate.odeint (LSODA adaptive solver)

</div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="font-family: IBM Plex Mono, monospace; font-size: 0.65rem; '
    'color: #334155; text-align: center;">'
    "Epidemic Spread Simulator · SIR Model · scipy + numpy + plotly + streamlit"
    "</div>",
    unsafe_allow_html=True,
)
