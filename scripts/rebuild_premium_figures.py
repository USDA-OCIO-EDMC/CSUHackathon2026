from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / "data" / "exports"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


COLORS = {
    "bg": "#f7f8fa",
    "ink": "#1f2937",
    "muted": "#6b7280",
    "accent": "#b91c1c",
    "accent2": "#0f766e",
    "accent3": "#1d4ed8",
    "line": "#d1d5db",
}


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "country": pd.read_csv(EXPORTS / "country_rollup.csv"),
        "lift": pd.read_csv(EXPORTS / "validation_tier_lift.csv"),
        "validation": pd.read_csv(EXPORTS / "validation_results.csv"),
        "holdout": pd.read_csv(EXPORTS / "validation_holdout_metrics.csv"),
    }


def save(fig: plt.Figure, out_name: str) -> Path:
    out = FIGURES / out_name
    fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def build_impact_funnel(country: pd.DataFrame) -> Path:
    # Core narrative numbers used in pitch artifacts
    total_routes = int(country["n_routes"].sum())
    top20_routes = int(round(total_routes * 0.20))
    top10_routes = int(round(total_routes * 0.10))
    signal20 = 60.8
    signal10 = 34.1

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")

    labels = [
        "All Scored Routes",
        "Top 20% Priority Set",
        "Top 10% Surge Set",
    ]
    widths = [1.00, 0.62, 0.36]
    counts = [total_routes, top20_routes, top10_routes]
    signals = [100.0, signal20, signal10]
    y = [2.3, 1.45, 0.6]
    colors = [COLORS["ink"], COLORS["accent2"], COLORS["accent"]]

    for yy, w, lab, c, n, s in zip(y, widths, labels, colors, counts, signals):
        rect = FancyBboxPatch(
            (0.08, yy),
            w,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0,
            facecolor=c,
            alpha=0.95,
        )
        ax.add_patch(rect)
        ax.text(0.11, yy + 0.33, lab, fontsize=16, color="white", fontweight="bold", va="center")
        ax.text(0.11, yy + 0.14, f"{n:,} routes", fontsize=12, color="white", va="center")
        ax.text(0.08 + w - 0.01, yy + 0.275, f"{s:.1f}% signal", fontsize=15, color="white", va="center", ha="right", fontweight="bold")

    ax.text(
        0.08,
        3.12,
        "Signal Capture Funnel",
        fontsize=26,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.08,
        2.95,
        "How route prioritization concentrates detection signal",
        fontsize=13,
        color=COLORS["muted"],
    )

    ax.text(
        0.08,
        0.08,
        "Top 20% covers 60.8% of historical detection signal. Top 10% captures 34.1%.",
        fontsize=12,
        color=COLORS["muted"],
        style="italic",
    )

    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 3.35)
    ax.axis("off")
    return save(fig, "impact_funnel_signal_capture.png")


def build_validation_scorecard(lift: pd.DataFrame, validation: pd.DataFrame, holdout: pd.DataFrame) -> Path:
    pearson = float(validation.loc[0, "pearson_r"])
    spearman = float(validation.loc[0, "spearman_r"])
    p10 = float(validation.loc[0, "precision_at_10"]) * 100
    holdout_p10 = float(holdout.loc[0, "precision_at_k"]) * 100
    holdout_pearson = float(holdout.loc[0, "pearson_r"])

    lift_map = lift.set_index("risk_tier")["mean_det"].to_dict()
    high_mean = float(lift_map.get("HIGH", 0.0))
    low_mean = float(lift_map.get("LOW", 1.0))
    lift_ratio = high_mean / max(low_mean, 1e-6)

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")

    cards = [
        ("Pearson r", f"{pearson:.3f}", "In-sample alignment", COLORS["accent3"]),
        ("Spearman r", f"{spearman:.3f}", "Rank consistency", COLORS["ink"]),
        ("Precision@10", f"{p10:.0f}%", "In-sample", COLORS["accent"]),
        ("Holdout P@10", f"{holdout_p10:.0f}%", "Temporal holdout", COLORS["accent2"]),
        ("Holdout Pearson", f"{holdout_pearson:.3f}", "Train ≤2022, test ≥2023", COLORS["ink"]),
        ("HIGH/LOW Lift", f"{lift_ratio:.2f}×", "Detection density ratio", COLORS["accent"]),
    ]

    w, h = 0.28, 0.26
    x0, y0 = 0.06, 0.56
    xgap, ygap = 0.035, 0.06
    for i, (title, value, subtitle, color) in enumerate(cards):
        row = i // 3
        col = i % 3
        x = x0 + col * (w + xgap)
        y = y0 - row * (h + ygap)
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1,
            edgecolor=COLORS["line"],
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.06, title, fontsize=12, color=COLORS["muted"], fontweight="bold")
        ax.text(x + 0.02, y + 0.11, value, fontsize=31, color=color, fontweight="bold")
        ax.text(x + 0.02, y + 0.04, subtitle, fontsize=10.5, color=COLORS["muted"])

    ax.text(0.06, 0.93, "Validation Scorecard", fontsize=26, fontweight="bold", color=COLORS["ink"])
    ax.text(0.06, 0.885, "Model quality snapshot from current export artifacts", fontsize=13, color=COLORS["muted"])
    ax.text(
        0.06,
        0.09,
        "Sources: validation_results.csv, validation_holdout_metrics.csv, validation_tier_lift.csv",
        fontsize=11,
        color=COLORS["muted"],
        style="italic",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return save(fig, "validation_scorecard_executive.png")


def build_country_priority_matrix(country: pd.DataFrame) -> Path:
    top = country.sort_values("max_risk", ascending=False).head(15).copy()
    top["label"] = top["origin_country"]

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")

    sizes = top["n_routes"].clip(lower=10) * 5
    sc = ax.scatter(
        top["mean_risk"],
        top["high_tier_routes"],
        s=sizes,
        c=top["max_risk"],
        cmap="Reds",
        alpha=0.85,
        edgecolor="white",
        linewidth=1.0,
    )

    for _, row in top.iterrows():
        ax.text(
            row["mean_risk"] + 0.25,
            row["high_tier_routes"] + 1.5,
            row["label"],
            fontsize=10.5,
            color=COLORS["ink"],
        )

    ax.set_title("Country Priority Matrix", fontsize=24, fontweight="bold", color=COLORS["ink"], pad=16)
    ax.text(
        0.01,
        1.02,
        "X = Mean risk score · Y = Number of HIGH-tier routes · Bubble size = route count",
        transform=ax.transAxes,
        fontsize=12,
        color=COLORS["muted"],
    )
    ax.set_xlabel("Mean Risk Score", fontsize=13, color=COLORS["ink"])
    ax.set_ylabel("HIGH-Tier Routes", fontsize=13, color=COLORS["ink"])
    ax.grid(True, color=COLORS["line"], linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Max Risk", fontsize=11, color=COLORS["ink"])

    ax.text(
        0.01,
        -0.12,
        "Source: country_rollup.csv (top 15 by max risk)",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["muted"],
        style="italic",
    )

    return save(fig, "country_priority_matrix.png")


def main():
    data = load_data()
    outputs = [
        build_impact_funnel(data["country"]),
        build_validation_scorecard(data["lift"], data["validation"], data["holdout"]),
        build_country_priority_matrix(data["country"]),
    ]
    print("Generated premium figures:")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()
