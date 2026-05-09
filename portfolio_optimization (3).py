# ============================================================
#  PORTFOLIO OPTIMIZATION VIA MONTE CARLO SIMULATION
#  Multi-Asset Mean-Variance Analysis | Efficient Frontier
# ============================================================
#  Assets  : SPY, QQQ, VTI, GLD, AGG
#  Lookback: 5 Years
#  Method  : Markowitz (1952) Mean-Variance Framework
#  Author  : [Your Name]  |  Course: FIN 4XX
# ============================================================

# ──────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yfinance as yf
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       130,
})

# ──────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────
TICKERS        = ["SPY", "QQQ", "VTI", "GLD", "AGG"]
RISK_FREE_RATE = 0.0525          # Current Fed Funds Rate proxy
N_SIMULATIONS  = 10_000          # Monte Carlo portfolios
TRADING_DAYS   = 252             # Annualisation factor
END_DATE       = datetime.today()
START_DATE     = END_DATE - timedelta(days=5 * 365)

# Asset colours (consistent across all charts)
ASSET_COLORS   = {
    "SPY": "#378ADD",
    "QQQ": "#D4537E",
    "VTI": "#1D9E75",
    "GLD": "#EF9F27",
    "AGG": "#7F77DD",
}

print("=" * 60)
print("  PORTFOLIO OPTIMIZATION — Monte Carlo Simulation")
print("=" * 60)
print(f"  Assets      : {', '.join(TICKERS)}")
print(f"  Period      : {START_DATE.date()}  →  {END_DATE.date()}")
print(f"  Simulations : {N_SIMULATIONS:,}")
print(f"  Risk-free r : {RISK_FREE_RATE*100:.2f}%")
print("=" * 60)

# ──────────────────────────────────────────
# 3. DATA DOWNLOAD
# ──────────────────────────────────────────
print("\n[1/5] Downloading price data from Yahoo Finance…")
raw   = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)["Close"]
data  = raw.dropna()
print(f"      ✓  {len(data)} trading days loaded  ({data.index[0].date()} – {data.index[-1].date()})")

# ──────────────────────────────────────────
# 4. RETURN STATISTICS
# ──────────────────────────────────────────
print("\n[2/5] Computing return statistics…")
daily_returns = data.pct_change().dropna()
mean_returns  = daily_returns.mean()          # μ  (daily)
cov_matrix    = daily_returns.cov()           # Σ  (daily)
corr_matrix   = daily_returns.corr()

ann_returns   = mean_returns * TRADING_DAYS
ann_vol       = daily_returns.std() * np.sqrt(TRADING_DAYS)

print("\n  Individual Asset Summary (Annualised):")
print(f"  {'Ticker':<8} {'Return':>8} {'Volatility':>12} {'Sharpe':>8}")
print("  " + "-" * 42)
for t in TICKERS:
    sr = (ann_returns[t] - RISK_FREE_RATE) / ann_vol[t]
    print(f"  {t:<8} {ann_returns[t]*100:>7.2f}%  {ann_vol[t]*100:>10.2f}%  {sr:>8.3f}")

# ──────────────────────────────────────────
# 5. MONTE CARLO SIMULATION
# ──────────────────────────────────────────
print(f"\n[3/5] Running {N_SIMULATIONS:,} Monte Carlo portfolio simulations…")

port_returns   = np.zeros(N_SIMULATIONS)
port_vols      = np.zeros(N_SIMULATIONS)
port_sharpes   = np.zeros(N_SIMULATIONS)
weights_matrix = np.zeros((N_SIMULATIONS, len(TICKERS)))

np.random.seed(42)   # reproducibility

for i in range(N_SIMULATIONS):
    # Dirichlet-style: exponential trick for uniform simplex sampling
    w  = np.random.exponential(1, len(TICKERS))
    w /= w.sum()
    weights_matrix[i] = w

    p_ret = np.dot(w, mean_returns) * TRADING_DAYS
    p_vol = np.sqrt(w @ cov_matrix.values @ w) * np.sqrt(TRADING_DAYS)
    p_sr  = (p_ret - RISK_FREE_RATE) / p_vol

    port_returns[i]  = p_ret
    port_vols[i]     = p_vol
    port_sharpes[i]  = p_sr

print("      ✓  Simulation complete")

# ──────────────────────────────────────────
# 6. IDENTIFY OPTIMAL PORTFOLIOS
# ──────────────────────────────────────────
print("\n[4/5] Identifying optimal portfolios…")

max_sharpe_idx = np.argmax(port_sharpes)
min_vol_idx    = np.argmin(port_vols)

ms_weights = weights_matrix[max_sharpe_idx]
mv_weights = weights_matrix[min_vol_idx]

# Equal-weight benchmark
ew_weights = np.full(len(TICKERS), 1 / len(TICKERS))
ew_ret  = np.dot(ew_weights, mean_returns) * TRADING_DAYS
ew_vol  = np.sqrt(ew_weights @ cov_matrix.values @ ew_weights) * np.sqrt(TRADING_DAYS)
ew_sr   = (ew_ret - RISK_FREE_RATE) / ew_vol

# ─── Print results ───
def print_portfolio(label, weights, ret, vol, sr):
    print(f"\n  ── {label} ──")
    print(f"  {'Metric':<20} {'Value':>10}")
    print("  " + "-" * 32)
    print(f"  {'Expected Return':<20} {ret*100:>9.2f}%")
    print(f"  {'Volatility':<20} {vol*100:>9.2f}%")
    print(f"  {'Sharpe Ratio':<20} {sr:>10.3f}")
    print(f"  {'Weights':}")
    for t, w in zip(TICKERS, weights):
        bar = "█" * int(w * 40)
        print(f"    {t:<6} {w*100:>5.1f}%  {bar}")

ms_ret = port_returns[max_sharpe_idx]
ms_vol = port_vols[max_sharpe_idx]
ms_sr  = port_sharpes[max_sharpe_idx]

mv_ret = port_returns[min_vol_idx]
mv_vol = port_vols[min_vol_idx]
mv_sr  = port_sharpes[min_vol_idx]

print_portfolio("MAX SHARPE PORTFOLIO",   ms_weights, ms_ret, ms_vol, ms_sr)
print_portfolio("MIN VOLATILITY PORTFOLIO", mv_weights, mv_ret, mv_vol, mv_sr)
print_portfolio("EQUAL-WEIGHT BENCHMARK",  ew_weights, ew_ret, ew_vol, ew_sr)

print(f"\n  Sharpe improvement vs equal-weight: "
      f"+{(ms_sr/ew_sr - 1)*100:.1f}%")

# ──────────────────────────────────────────
# 7. BUILD EFFICIENT FRONTIER CURVE
# ──────────────────────────────────────────
sorted_idx = np.argsort(port_vols)
sorted_vols = port_vols[sorted_idx]
sorted_rets = port_returns[sorted_idx]

frontier_vols, frontier_rets = [sorted_vols[0]], [sorted_rets[0]]
max_ret_so_far = sorted_rets[0]
for v, r in zip(sorted_vols[1:], sorted_rets[1:]):
    if r > max_ret_so_far:
        max_ret_so_far = r
        frontier_vols.append(v)
        frontier_rets.append(r)

# ──────────────────────────────────────────
# 8. VISUALISATION  (7 separate figures)
# ──────────────────────────────────────────
print("\n[5/5] Generating 7 individual charts…")

BG   = "#FAFAF9"
SUBTITLE = (f"SPY · QQQ · VTI · GLD · AGG  |  5-Year Lookback  |  "
            f"{N_SIMULATIONS:,} Simulations  |  Risk-Free Rate: {RISK_FREE_RATE*100:.2f}%")

def add_titles(fig, title, subtitle=SUBTITLE):
    fig.text(0.5, 0.97, title,    ha="center", va="top",
             fontsize=14, fontweight="bold",  color="#1a1a1a")
    fig.text(0.5, 0.93, subtitle, ha="center", va="top",
             fontsize=9,  color="#666666")

saved = []

# ─────────────────────────────────────────
# FIGURE 1 — Efficient Frontier
# ─────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 7))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
fig1.subplots_adjust(top=0.86)

sc = ax1.scatter(
    port_vols * 100, port_returns * 100,
    c=port_sharpes, cmap="viridis", s=5, alpha=0.55, linewidths=0,
)
cbar = fig1.colorbar(sc, ax=ax1, pad=0.02)
cbar.set_label("Sharpe Ratio", fontsize=10, color="#555")
cbar.ax.tick_params(labelsize=9)

ax1.plot([v * 100 for v in frontier_vols],
         [r * 100 for r in frontier_rets],
         color="#EF9F27", lw=2.5, label="Efficient Frontier", zorder=5)
ax1.scatter(ms_vol*100, ms_ret*100, color="#E24B4A", s=220, zorder=10,
            label=f"Max Sharpe  (SR = {ms_sr:.2f})", edgecolors="white", lw=1.5)
ax1.scatter(mv_vol*100, mv_ret*100, color="#1D9E75", s=220, zorder=10,
            label=f"Min Volatility  ({mv_vol*100:.1f}% vol)", edgecolors="white", lw=1.5)
ax1.scatter(ew_vol*100, ew_ret*100, color="#888780", s=160, zorder=10,
            label=f"Equal Weight  (SR = {ew_sr:.2f})",
            edgecolors="white", lw=1.5, marker="D")
ax1.set_xlabel("Annualised Volatility (%)", fontsize=11)
ax1.set_ylabel("Annualised Return (%)",     fontsize=11)
ax1.set_title("Efficient Frontier — Risk vs. Return Space", fontsize=12, pad=10)
ax1.legend(fontsize=10, framealpha=0.88, edgecolor="#ddd")
add_titles(fig1, "Figure 1 — Efficient Frontier")
p = "fig1_efficient_frontier.png"
fig1.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
saved.append(p); print(f"  ✓  Saved {p}")
plt.show()

# ─────────────────────────────────────────
# FIGURE 2 — Max Sharpe Portfolio Weights
# ─────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)
fig2.subplots_adjust(top=0.84, left=0.12, right=0.88)

sorted_ms = sorted(zip(TICKERS, ms_weights), key=lambda x: x[1])
t_ms, w_ms = zip(*sorted_ms)
bars2 = ax2.barh(t_ms, [w*100 for w in w_ms],
                 color=[ASSET_COLORS[t] for t in t_ms],
                 edgecolor="white", height=0.55)
for bar, w in zip(bars2, w_ms):
    ax2.text(bar.get_width() + 0.4,
             bar.get_y() + bar.get_height() / 2,
             f"{w*100:.1f}%", va="center", fontsize=11, color="#333")
ax2.set_xlabel("Allocation (%)", fontsize=11)
ax2.set_xlim(0, max(w_ms) * 130)
ax2.set_title(f"Max Sharpe Portfolio  |  Expected Return: {ms_ret*100:.1f}%  "
              f"|  Volatility: {ms_vol*100:.1f}%  |  Sharpe: {ms_sr:.2f}",
              fontsize=10, pad=8)
add_titles(fig2, "Figure 2 — Optimal Portfolio Weights (Max Sharpe)")
p = "fig2_max_sharpe_weights.png"
fig2.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
saved.append(p); print(f"  ✓  Saved {p}")
plt.show()

# ─────────────────────────────────────────
# FIGURE 3 — Min Volatility Portfolio Weights
# ─────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.patch.set_facecolor(BG)
ax3.set_facecolor(BG)
fig3.subplots_adjust(top=0.84, left=0.12, right=0.88)

sorted_mv = sorted(zip(TICKERS, mv_weights), key=lambda x: x[1])
t_mv, w_mv = zip(*sorted_mv)
bars3 = ax3.barh(t_mv, [w*100 for w in w_mv],
                 color=[ASSET_COLORS[t] for t in t_mv],
                 edgecolor="white", height=0.55)
for bar, w in zip(bars3, w_mv):
    ax3.text(bar.get_width() + 0.4,
             bar.get_y() + bar.get_height() / 2,
             f"{w*100:.1f}%", va="center", fontsize=11, color="#333")
ax3.set_xlabel("Allocation (%)", fontsize=11)
ax3.set_xlim(0, max(w_mv) * 130)
ax3.set_title(f"Min Volatility Portfolio  |  Expected Return: {mv_ret*100:.1f}%  "
              f"|  Volatility: {mv_vol*100:.1f}%  |  Sharpe: {mv_sr:.2f}",
              fontsize=10, pad=8)
add_titles(fig3, "Figure 3 — Optimal Portfolio Weights (Min Volatility)")
p = "fig3_min_vol_weights.png"
fig3.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
saved.append(p); print(f"  ✓  Saved {p}")
plt.show()

# ─────────────────────────────────────────
# FIGURE 4 — Correlation Heatmap
# ─────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 6))
fig4.patch.set_facecolor(BG)
fig4.subplots_adjust(top=0.84)

sns.heatmap(
    corr_matrix, ax=ax4, annot=True, fmt=".2f",
    cmap="RdBu_r", vmin=-1, vmax=1,
    square=True, linewidths=0.8, linecolor="#eee",
    cbar_kws={"shrink": 0.82, "pad": 0.03},
    annot_kws={"size": 12, "weight": "bold"},
)
ax4.set_title("Pairwise Return Correlations (Daily)", fontsize=11, pad=10)
ax4.tick_params(axis="x", rotation=0,  labelsize=11)
ax4.tick_params(axis="y", rotation=0,  labelsize=11)
add_titles(fig4, "Figure 4 — Correlation Matrix")
p = "fig4_correlation_matrix.png"
fig4.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
saved.append(p); print(f"  ✓  Saved {p}")
plt.show()

# ─────────────────────────────────────────
# FIGURE 5 — Portfolio Return Distribution
# ─────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(10, 5))
fig5.patch.set_facecolor(BG)
ax5.set_facecolor(BG)
fig5.subplots_adjust(top=0.84)

ax5.hist(port_returns * 100, bins=70, color="#378ADD", edgecolor="none", alpha=0.85)
ax5.axvline(ms_ret * 100, color="#E24B4A", lw=2.2,
            label=f"Max Sharpe:  {ms_ret*100:.2f}%")
ax5.axvline(mv_ret * 100, color="#1D9E75", lw=2.2,
            label=f"Min Vol:       {mv_ret*100:.2f}%")
ax5.axvline(ew_ret * 100, color="#888780", lw=2,   linestyle="--",
            label=f"Equal Weight: {ew_ret*100:.2f}%")
ax5.set_xlabel("Annualised Return (%)", fontsize=11)
ax5.set_ylabel("Number of Portfolios",  fontsize=11)
ax5.set_title("Distribution of Simulated Portfolio Returns", fontsize=12, pad=8)
ax5.legend(fontsize=10, framealpha=0.88, edgecolor="#ddd")
add_titles(fig5, "Figure 5 — Portfolio Return Distribution")
p = "fig5_return_distribution.png"
fig5.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
saved.append(p); print(f"  ✓  Saved {p}")
plt.show()

# ─────────────────────────────────────────
# FIGURE 6 — Sharpe Ratio Distribution
# ─────────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(10, 5))
fig6.patch.set_facecolor(BG)
ax6.set_facecolor(BG)
fig6.subplots_adjust(top=0.84)

ax6.hist(port_sharpes, bins=70, color="#D4537E", edgecolor="none", alpha=0.85)
ax6.axvline(ms_sr, color="#E24B4A", lw=2.2,
            label=f"Max Sharpe:   {ms_sr:.3f}")
ax6.axvline(ew_sr, color="#888780", lw=2,   linestyle="--",
            label=f"Equal Weight: {ew_sr:.3f}")
ax6.set_xlabel("Sharpe Ratio",          fontsize=11)
ax6.set_ylabel("Number of Portfolios",  fontsize=11)
ax6.set_title("Distribution of Simulated Sharpe Ratios", fontsize=12, pad=8)
ax6.legend(fontsize=10, framealpha=0.88, edgecolor="#ddd")
add_titles(fig6, "Figure 6 — Sharpe Ratio Distribution")
p = "fig6_sharpe_distribution.png"
fig6.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
saved.append(p); print(f"  ✓  Saved {p}")
plt.show()

# ─────────────────────────────────────────
# FIGURE 7 — Individual Asset Risk / Return
# ─────────────────────────────────────────
fig7, ax7 = plt.subplots(figsize=(9, 6))
fig7.patch.set_facecolor(BG)
ax7.set_facecolor(BG)
fig7.subplots_adjust(top=0.84)

for t in TICKERS:
    ax7.scatter(ann_vol[t]*100, ann_returns[t]*100,
                color=ASSET_COLORS[t], s=180, zorder=5,
                edgecolors="white", lw=1.5)
    ax7.annotate(t,
                 xy=(ann_vol[t]*100, ann_returns[t]*100),
                 xytext=(6, 5), textcoords="offset points",
                 fontsize=11, color=ASSET_COLORS[t], fontweight="bold")

ax7.scatter(ms_vol*100, ms_ret*100, color="#E24B4A", s=250, marker="*",
            zorder=10, label="Max Sharpe Portfolio",
            edgecolors="white", lw=1)
ax7.set_xlabel("Annualised Volatility (%)", fontsize=11)
ax7.set_ylabel("Annualised Return (%)",     fontsize=11)
ax7.set_title("Individual Asset Risk vs. Return (Annualised)", fontsize=12, pad=8)
legend_patches = [mpatches.Patch(color=ASSET_COLORS[t], label=t) for t in TICKERS]
legend_patches.append(mpatches.Patch(color="#E24B4A", label="Max Sharpe Portfolio"))
ax7.legend(handles=legend_patches, fontsize=10, framealpha=0.88,
           edgecolor="#ddd", ncol=2)
add_titles(fig7, "Figure 7 — Individual Asset Risk / Return")
p = "fig7_asset_risk_return.png"
fig7.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
saved.append(p); print(f"  ✓  Saved {p}")
plt.show()

print(f"\n  All {len(saved)} charts saved to your working directory.")

# ──────────────────────────────────────────
# 9. SUMMARY TABLE
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  PORTFOLIO COMPARISON SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    "Max Sharpe":     [f"{ms_ret*100:.2f}%",  f"{ms_vol*100:.2f}%",  f"{ms_sr:.3f}"],
    "Min Volatility": [f"{mv_ret*100:.2f}%",  f"{mv_vol*100:.2f}%",  f"{mv_sr:.3f}"],
    "Equal Weight":   [f"{ew_ret*100:.2f}%",  f"{ew_vol*100:.2f}%",  f"{ew_sr:.3f}"],
}, index=["Expected Return (p.a.)", "Volatility (p.a.)", "Sharpe Ratio"])
print(summary.to_string())

print("\n  Max Sharpe Optimal Weights:")
for t, w in sorted(zip(TICKERS, ms_weights), key=lambda x: -x[1]):
    print(f"    {t:<6} {w*100:>6.2f}%")

print(f"\n  Sharpe ratio gain over equal-weight: +{(ms_sr/ew_sr-1)*100:.1f}%")
print("\n" + "=" * 60)
print("  Analysis complete.")
print("=" * 60)
