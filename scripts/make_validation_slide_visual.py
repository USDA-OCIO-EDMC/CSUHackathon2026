from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / "data" / "exports"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

OUT = FIGURES / "validation_graph_matrix_slide.png"


def main() -> None:
    ports = pd.read_csv(EXPORTS / "validation_port_table.csv")
    val = pd.read_csv(EXPORTS / "validation_results.csv").iloc[0]
    hold = pd.read_csv(EXPORTS / "validation_holdout_metrics.csv").iloc[0]
    lift = pd.read_csv(EXPORTS / "validation_tier_lift.csv")
    sens = pd.read_csv(EXPORTS / "validation_sensitivity.csv").iloc[0]

    pearson = float(val["pearson_r"])
    p10 = float(val["precision_at_10"]) * 100
    holdout_p10 = float(hold["precision_at_k"]) * 100
    holdout_pearson = float(hold["pearson_r"])
    lift_map = lift.set_index("risk_tier")["mean_det"].to_dict()
    high_det = float(lift_map.get("HIGH", 0))
    med_det = float(lift_map.get("MEDIUM", 0))
    low_det = float(lift_map.get("LOW", 1e-6))
    lift_ratio = high_det / max(low_det, 1e-9)

    # Styling
    plt.rcParams["font.family"] = "DejaVu Sans"
    bg = "#f8fafc"
    ink = "#111827"
    muted = "#6b7280"
    accent = "#b91c1c"
    accent2 = "#0f766e"
    accent3 = "#1d4ed8"
    line = "#d1d5db"

    fig = plt.figure(figsize=(16, 9), dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        nrows=8,
        ncols=12,
        left=0.04,
        right=0.98,
        top=0.92,
        bottom=0.08,
        wspace=0.8,
        hspace=0.8,
    )

    # Title area
    fig.text(
        0.04,
        0.955,
        "Validated Against Five Years of APHIS Detections",
        fontsize=28,
        fontweight="bold",
        color=ink,
    )
    fig.text(
        0.04,
        0.925,
        "Port-level alignment + operational lift + temporal holdout",
        fontsize=12.5,
        color=muted,
    )

    # Left: scatter plot
    ax_sc = fig.add_subplot(gs[:, :7])
    ax_sc.set_facecolor(bg)

    x = ports["mean_risk"].to_numpy()
    y = ports["detection_count"].to_numpy()
    ax_sc.scatter(x, y, s=90, color=accent3, alpha=0.85, edgecolor="white", linewidth=1.0)

    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() - 1, x.max() + 1, 200)
        ys = m * xs + b
        ax_sc.plot(xs, ys, color=accent, linewidth=2.2, alpha=0.9)

    # Label top detection ports
    top_ports = ports.sort_values("detection_count", ascending=False).head(8)
    for _, r in top_ports.iterrows():
        ax_sc.annotate(
            str(r["us_port"]),
            (r["mean_risk"], r["detection_count"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9.5,
            color=ink,
        )

    ax_sc.set_title(
        f"Port Risk vs Detection Count  (Pearson r = {pearson:.3f})",
        fontsize=14.5,
        color=ink,
        pad=10,
        fontweight="bold",
    )
    ax_sc.set_xlabel("Mean Port Risk Score", fontsize=12, color=ink)
    ax_sc.set_ylabel("APHIS Detection Count", fontsize=12, color=ink)
    ax_sc.grid(True, color=line, linewidth=0.8, alpha=0.75)
    ax_sc.tick_params(colors=muted)
    for spine in ax_sc.spines.values():
        spine.set_color(line)

    # Right-top: metrics matrix
    ax_mt = fig.add_subplot(gs[:4, 7:])
    ax_mt.axis("off")
    ax_mt.set_facecolor("white")

    cards = [
        ("Pearson r", f"{pearson:.3f}", "in-sample", accent3),
        ("Precision@10", f"{p10:.0f}%", "in-sample", accent),
        ("Holdout Pearson", f"{holdout_pearson:.3f}", "train ≤2022 / test ≥2023", ink),
        ("Holdout P@10", f"{holdout_p10:.0f}%", "future generalization", accent2),
    ]

    card_w, card_h = 0.45, 0.4
    positions = [(0.02, 0.52), (0.53, 0.52), (0.02, 0.05), (0.53, 0.05)]
    for (title, value, subtitle, col), (x0, y0) in zip(cards, positions):
        patch = FancyBboxPatch(
            (x0, y0),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=line,
            facecolor=bg,
            transform=ax_mt.transAxes,
        )
        ax_mt.add_patch(patch)
        ax_mt.text(x0 + 0.03, y0 + 0.30, title, fontsize=11, color=muted, fontweight="bold", transform=ax_mt.transAxes)
        ax_mt.text(x0 + 0.03, y0 + 0.15, value, fontsize=27, color=col, fontweight="bold", transform=ax_mt.transAxes)
        ax_mt.text(x0 + 0.03, y0 + 0.05, subtitle, fontsize=9.5, color=muted, transform=ax_mt.transAxes)

    # Right-mid: tier lift bars
    ax_lift = fig.add_subplot(gs[4:7, 7:])
    ax_lift.set_facecolor(bg)
    tiers = ["LOW", "MEDIUM", "HIGH"]
    vals = [low_det, med_det, high_det]
    colors = [muted, accent2, accent]
    bars = ax_lift.barh(tiers, vals, color=colors, alpha=0.9)
    ax_lift.set_title(f"Detection Density by Tier  (HIGH/LOW = {lift_ratio:.2f}x)", fontsize=12.5, color=ink, fontweight="bold")
    ax_lift.grid(axis="x", color=line, linewidth=0.8, alpha=0.75)
    ax_lift.tick_params(colors=muted)
    for spine in ax_lift.spines.values():
        spine.set_color(line)
    for bar, valv in zip(bars, vals):
        ax_lift.text(valv + max(vals) * 0.03, bar.get_y() + bar.get_height() / 2, f"{valv:.2f}", va="center", fontsize=10.5, color=ink)

    # Bottom strip: caveat + leakage line
    ax_note = fig.add_subplot(gs[7:, :])
    ax_note.axis("off")
    ax_note.text(
        0.0,
        0.65,
        "Surveillance bias disclosed: detections reflect where inspectors trapped, not necessarily where flies arrived.",
        fontsize=11.5,
        color=muted,
        style="italic",
        transform=ax_note.transAxes,
    )
    ax_note.text(
        0.0,
        0.20,
        f"Leakage sensitivity: Pearson drops from {float(sens['full_pearson_r']):.3f} to "
        f"{float(sens['no_detection_pearson_r']):.3f}, while Precision@10 remains "
        f"{float(sens['full_precision_at_k']) * 100:.0f}%.",
        fontsize=11,
        color=muted,
        transform=ax_note.transAxes,
    )

    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
