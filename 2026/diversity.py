"""
Analyze race-bias hate crimes per capita vs. neighbourhood diversity.

Steps:
  1. Verify diversity_df has 158 unique neighbourhoods.
  2. Count RACE_BIAS incidents per neighbourhood (HOOD_158) in crimes_df.
  3. Merge with diversity_df to get population (Total) and minority % per neighbourhood.
  4. Compute per-capita race-bias incidents.
  5. OLS regression: per-capita incidents ~ Percent Of Minority.
  6. Scatter plot with regression line, confidence interval, and annotated stats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import warnings
import geopandas as gpd
import contextily as ctx
warnings.filterwarnings("ignore")

# Helper — draw one choropleth
# ══════════════════════════════════════════════════════════════════════════════
def _choropleth(gdf_plot, column, title, cbar_label, cmap, out_file,
                label_col=None):
    """
    label_col : column whose value is printed at each polygon's centroid.
                Pass SHP_JOIN_COL (neighbourhood number) to show hood labels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Grey fill for unmatched polygons
    gdf_plot[gdf_plot[column].isna()].plot(
        ax=ax, color="#cccccc", edgecolor="white", linewidth=0.3
    )

    # Choropleth layer
    gdf_plot[gdf_plot[column].notna()].plot(
        column=column,
        ax=ax,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.80,
        legend=True,
        legend_kwds={
            "label": cbar_label,
            "orientation": "vertical",
            "shrink": 0.6,
            "pad": 0.02,
        }
    )

    # Basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # ── Neighbourhood number labels at polygon centroids ──────────────────────
    if label_col:
        # Use representative_point() instead of centroid to keep label
        # inside irregular/concave polygons
        label_gdf = gdf_plot[[label_col, "geometry"]].copy()
        label_gdf["_pt"] = label_gdf.geometry.representative_point()

        for _, row in label_gdf.iterrows():
            val = row[label_col]
            if pd.isna(val):
                continue
            ax.annotate(
                text=str(int(val)),
                xy=(row["_pt"].x, row["_pt"].y),
                ha="center", va="center",
                fontsize=   5.5,
                fontweight="bold",
                color="black",
                # Thin white halo so the number is legible on any fill colour
                path_effects=[
                    __import__("matplotlib.patheffects", fromlist=["withStroke"])
                    .withStroke(linewidth=2, foreground="white")
                ],
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out_file}")


if __name__ == "__main__":

    crimes_df   = pd.read_csv("data/hatecrimes.csv")
    diversity_df = pd.read_csv("data/minority.csv")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 1 — Verify diversity_df has 158 unique neighbourhoods
    # ══════════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 1 — Neighbourhood count check")
    print("=" * 60)

    n_unique = diversity_df["Neighbourhood Number"].nunique()
    n_rows   = len(diversity_df)

    print(f"  Unique neighbourhood IDs : {n_unique}")
    print(f"  Total rows in diversity_df: {n_rows}")

    if n_unique == 158:
        print("  ✓ Confirmed: 158 unique neighbourhoods.\n")
    else:
        print(f"  ✗ WARNING: Expected 158, found {n_unique}. "
              "Check for duplicates or missing entries.\n")
        dupes = diversity_df[diversity_df.duplicated("Neighbourhood Number", keep=False)]
        if not dupes.empty:
            print("  Duplicate neighbourhood IDs:")
            print(dupes[["Neighbourhood Number"]].to_string(index=True))
            print()

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 2 — Count RACE_BIAS incidents per neighbourhood
    # ══════════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 2 — Race-bias incident counts")
    print("=" * 60)

    # Keep only race-bias rows; treat NaN / blank as non-race-bias
    race_bias_df = crimes_df[
        crimes_df["RACE_BIAS"].notna() &
        (crimes_df["RACE_BIAS"].astype(str).str.strip() != "") &
        (crimes_df["RACE_BIAS"].astype(str).str.upper() != "NONE")
    ].copy()

    incident_counts = (
        race_bias_df
        .groupby("HOOD_158", as_index=False)
        .size()
        .rename(columns={"HOOD_158": "Neighbourhood Number", "size": "race_bias_count"})
    )

    print(f"  Total race-bias incidents : {race_bias_df.shape[0]:,}")
    print(f"  Neighbourhoods with ≥1 incident: {len(incident_counts)}\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 3 — Merge with diversity_df
    # ══════════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 3 — Merge & per-capita calculation")
    print("=" * 60)

    # Ensure join keys are the same dtype
    diversity_df["Neighbourhood Number"] = pd.to_numeric(
        diversity_df["Neighbourhood Number"], errors="coerce"
    )
    incident_counts["Neighbourhood Number"] = pd.to_numeric(
        incident_counts["Neighbourhood Number"], errors="coerce"
    )


    merged = diversity_df.merge(incident_counts, on="Neighbourhood Number", how="left")
    merged["race_bias_count"] = merged["race_bias_count"].fillna(0)
    merged['Percent Of Minority'] = merged["  Total visible minority population"] / merged[
        'Total']

    # Clean population column
    merged["Total"] = pd.to_numeric(
        merged["Total"].astype(str).str.replace(",", ""), errors="coerce"
    )
    merged["Percent Of Minority"] = pd.to_numeric(
        merged["Percent Of Minority"].astype(str).str.replace("%", "").str.strip(),
        errors="coerce"
    )

    # Drop rows where population is missing or zero (can't compute per capita)
    before = len(merged)
    merged = merged[merged["Total"].notna() & (merged["Total"] > 0)].copy()
    after  = len(merged)
    if before != after:
        print(f"  ⚠ Dropped {before - after} row(s) with missing/zero population.\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 4 — Compute per-capita rate (incidents per 10,000 residents)
    # ══════════════════════════════════════════════════════════════════════════════
    SCALE = 10_000  # incidents per 10,000 residents
    merged["race_bias_per_capita"] = (merged["race_bias_count"] / merged["Total"]) * SCALE

    print(f"  Neighbourhoods in analysis : {len(merged)}")
    print(f"  Per-capita metric          : incidents per {SCALE:,} residents")
    print(f"  Mean per-capita rate       : {merged['race_bias_per_capita'].mean():.4f}")
    print(f"  Max  per-capita rate       : {merged['race_bias_per_capita'].max():.4f}")
    print()

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 5a — Initial OLS (full data) to identify high-influence points
    # ══════════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 5a — Initial OLS (full data) — identifying outliers")
    print("=" * 60)

    reg_df = merged[["Neighbourhood Number", "Percent Of Minority",
                     "race_bias_per_capita"]].dropna().copy().reset_index(drop=True)

    X_full = sm.add_constant(reg_df["Percent Of Minority"])
    y_full = reg_df["race_bias_per_capita"]

    model_full = sm.OLS(y_full, X_full).fit()
    influence_full = OLSInfluence(model_full)
    cooks_d_full = influence_full.cooks_distance[0]
    outlier_threshold = 4 / int(model_full.nobs)
    outlier_idx = np.where(cooks_d_full > outlier_threshold)[0]

    outlier_hoods = reg_df.loc[outlier_idx, "Neighbourhood Number"].tolist()

    print(f"  Cook's D threshold (4/n)  : {outlier_threshold:.4f}")
    print(f"  High-influence points     : {len(outlier_idx)}")
    print(f"  Neighbourhood Numbers     : {sorted([int(h) for h in outlier_hoods])}\n")

    if len(outlier_idx):
        print("  Detail of removed neighbourhoods:")
        print(f"  {'Hood #':<10} {'% Minority':>12} {'Incidents/10k':>15} {'Cook\\s D':>12}")
        print("  " + "-" * 52)
        for i in outlier_idx:
            row = reg_df.iloc[i]
            print(f"  {int(row['Neighbourhood Number']):<10} "
              f"{row['Percent Of Minority']:>11.1f}% "
              f"{row['race_bias_per_capita']:>15.4f} "
              f"{cooks_d_full[i]:>12.4f}")
        print()

        # ══════════════════════════════════════════════════════════════════════════════
        # STEP 5b — Re-run OLS with outliers removed
        # ══════════════════════════════════════════════════════════════════════════════
        print("=" * 60)
        print("STEP 5b — OLS (outliers removed)")
        print("=" * 60)

        reg_df_clean = reg_df.drop(index=outlier_idx).reset_index(drop=True)

        X_clean = sm.add_constant(reg_df_clean["Percent Of Minority"])
        y_clean = reg_df_clean["race_bias_per_capita"]

        model = sm.OLS(y_clean, X_clean).fit()
        print(model.summary())
        print()

        beta0 = model.params["const"]
        beta1 = model.params["Percent Of Minority"]
        p_val_b1 = model.pvalues["Percent Of Minority"]
        r_squared = model.rsquared
        n_obs = int(model.nobs)

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 6 — Plot
    # ══════════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 5b — OLS (outliers removed)")
    print("=" * 60)

    reg_df_clean = reg_df.drop(index=outlier_idx).reset_index(drop=True)

    X_clean = sm.add_constant(reg_df_clean["Percent Of Minority"])
    y_clean = reg_df_clean["race_bias_per_capita"]

    model = sm.OLS(y_clean, X_clean).fit()

    beta0 = model.params["const"]
    beta1 = model.params["Percent Of Minority"]
    p_val_b1 = model.pvalues["Percent Of Minority"]
    r_squared = model.rsquared
    n_obs = int(model.nobs)

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 6 — Plot (cleaned data + removed outliers shown separately)
    # ══════════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))

    # ── Clean points ─────────────────────────────────────────────────────────────
    ax.scatter(
        reg_df_clean["Percent Of Minority"],
        reg_df_clean["race_bias_per_capita"],
        c="steelblue", alpha=0.65, edgecolors="white", linewidths=0.4,
        s=60, zorder=3, label="Neighbourhood (in regression)"
    )

    # ── Removed outliers (plotted but excluded from fit) ─────────────────────────
    if len(outlier_idx):
        outlier_rows = reg_df.iloc[outlier_idx]
        ax.scatter(
            outlier_rows["Percent Of Minority"],
            outlier_rows["race_bias_per_capita"],
            c="tomato", alpha=0.75, edgecolors="darkred", linewidths=0.8,
            s=80, marker="D", zorder=4,
            label=f"Removed (Cook's D > {outlier_threshold:.3f})"
        )
        # Label each outlier with its neighbourhood number
        for _, row in outlier_rows.iterrows():
            ax.annotate(
                f"#{int(row['Neighbourhood Number'])}",
                xy=(row["Percent Of Minority"], row["race_bias_per_capita"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8, color="darkred"
            )

    # ── Regression line + 95% CI (clean model) ───────────────────────────────────
    x_range = np.linspace(reg_df_clean["Percent Of Minority"].min(),
                          reg_df_clean["Percent Of Minority"].max(), 200)
    x_pred = sm.add_constant(x_range)
    pred_df = model.get_prediction(x_pred).summary_frame(alpha=0.05)

    ax.plot(x_range, pred_df["mean"],
            color="steelblue", linewidth=2, zorder=5, label="OLS fit (cleaned)")
    ax.fill_between(x_range,
                    pred_df["mean_ci_lower"], pred_df["mean_ci_upper"],
                    color="steelblue", alpha=0.15, label="95% confidence interval")

    # ── Annotation box ────────────────────────────────────────────────────────────
    sig_star = (
        "***" if p_val_b1 < 0.001 else
        "**" if p_val_b1 < 0.01 else
        "*" if p_val_b1 < 0.05 else
        "ns"
    )
    stats_text = (
        f"β₁ = {beta1:.4f}{sig_star}\n"
        f"p  = {p_val_b1:.4f}\n"
        f"R² = {r_squared:.4f}\n"
        f"n  = {n_obs} (excl. {len(outlier_idx)} outlier{'s' if len(outlier_idx) != 1 else ''})"
    )
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray", alpha=0.9))

    # ── Labels & styling ──────────────────────────────────────────────────────────
    ax.set_xlabel("Percent of Minority (%)", fontsize=12)
    ax.set_ylabel(f"Race-Bias Incidents per {SCALE:,} Residents", fontsize=12)
    ax.set_title(
        "Race-Bias Hate Crime Rate vs. Neighbourhood Diversity\n"
        "(Toronto Neighbourhoods — high-influence points removed)",
        fontsize=13, fontweight="bold", pad=12
    )

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.show()

    # ══════════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PER-CAPITA RACE-BIAS RATE BY NEIGHBOURHOOD (top 20)")
    print("=" * 60)
    summary = (
        merged[["Neighbourhood Number", "race_bias_count",
                "Total", "Percent Of Minority", "race_bias_per_capita"]]
        .sort_values("race_bias_per_capita", ascending=False)
        .head(20)
        .rename(columns={
            "Neighbourhood Number": "Hood #",
            "race_bias_count": "Incidents",
            "Total": "Population",
            "Percent Of Minority": "% Minority",
            "race_bias_per_capita": f"Incidents/10k"
        })
    )
    print(summary.to_string(index=False, float_format="%.2f"))

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 7 — Load shapefile & merge with analysis data
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 7 — Choropleth maps")
    print("=" * 60)

    gdf = gpd.read_file("data/Neighbourhoods - 4326.shp")
    print(f"  Shapefile CRS      : {gdf.crs}")
    print(f"  Shapefile columns  : {list(gdf.columns)}")
    print(f"  Shapefile row count: {len(gdf)}")

    pd.set_option("display.max_columns", None)  # show all columns
    pd.set_option("display.width", None)  # don't wrap to fit terminal width
    pd.set_option("display.max_colwidth", None)  # don't truncate cell contents

    # ── Identify the neighbourhood-number column in the shapefile ─────────────────
    # Common names used by Toronto Open Data shapefiles:
    _candidate_cols = ["AREA_SH5"]

    if not _candidate_cols:
        # Fall back: show all columns and let the user pick
        raise ValueError(
            "Could not auto-detect the neighbourhood-number column in the shapefile.\n"
            f"Available columns: {list(gdf.columns)}\n"
            "Set SHP_JOIN_COL below manually."
        )

    SHP_JOIN_COL = _candidate_cols[0]  # ← override here if needed
    print(f"  Joining on shapefile column: '{SHP_JOIN_COL}'")

    gdf[SHP_JOIN_COL] = pd.to_numeric(gdf[SHP_JOIN_COL], errors="coerce")

    # Reproject to Web Mercator (required by contextily)
    gdf = gdf.to_crs(epsg=3857)

    # Merge analysis data onto geodataframe
    plot_gdf = gdf.merge(
        merged[["Neighbourhood Number", "Percent Of Minority", "race_bias_per_capita"]],
        left_on=SHP_JOIN_COL,
        right_on="Neighbourhood Number",
        how="left"
    )

    missing = plot_gdf["Percent Of Minority"].isna().sum()
    if missing:
        print(f"  ⚠  {missing} shapefile polygon(s) had no match in merged data "
              "(will appear grey on maps).")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 8a — Choropleth: Minority % (diversity)
    # ══════════════════════════════════════════════════════════════════════════════
    _choropleth(
        gdf_plot=plot_gdf,
        column="Percent Of Minority",
        title="Neighbourhood Diversity — Visible Minority Population (%)\nToronto",
        cbar_label="Visible Minority (%)",
        cmap="YlOrRd",
        label_col  = SHP_JOIN_COL,
    )

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 8b — Choropleth: Race-bias incidents per 10,000 residents
    # ══════════════════════════════════════════════════════════════════════════════
    _choropleth(
        gdf_plot=plot_gdf,
        column="race_bias_per_capita",
        title=f"Race-Bias Hate Crimes per {SCALE:,} Residents\nToronto Neighbourhoods",
        cbar_label=f"Incidents per {SCALE:,} residents",
        cmap="Blues",
        label_col  = SHP_JOIN_COL,
    )