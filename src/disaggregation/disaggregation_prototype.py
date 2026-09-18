"""
Prototype: Mass-preserving statistical disaggregation of district-level
disease case counts to fine-resolution (H3-like) cells using covariates.

This is a synthetic-data proof of concept for the Dengue/Malaria Research
Blueprint. It demonstrates the core method (Poisson regression with an
aggregation constraint, in the spirit of the Malaria Atlas Project's
`disaggregation` approach) before we touch any real data.

Steps:
  1. Simulate a "world" of districts, each split into many fine cells.
  2. Simulate a HIDDEN true risk surface driven by covariates (population,
     NDVI-like vegetation, rainfall) -- this stands in for the fine-resolution
     truth we could never observe in real surveillance data.
  3. Aggregate the hidden fine-cell case counts up to district totals --
     this simulates what real district-level surveillance data looks like.
  4. Fit a disaggregation model using ONLY district totals + covariates
     (never touching the hidden truth).
  5. Validate: compare recovered fine-cell predictions against the hidden
     truth to see how well the method recovers the real spatial pattern.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Simulate the world: districts subdivided into fine (H3-like) cells
# ----------------------------------------------------------------------

N_DISTRICTS_X = 15
N_DISTRICTS_Y = 15
CELLS_PER_DISTRICT_SIDE = 6  # each district is a CELLS_PER_DISTRICT_SIDE x same grid of fine cells

records = []
district_id = 0
for dx in range(N_DISTRICTS_X):
    for dy in range(N_DISTRICTS_Y):
        district_id += 1
        for cx in range(CELLS_PER_DISTRICT_SIDE):
            for cy in range(CELLS_PER_DISTRICT_SIDE):
                # global fine-cell coordinates (used only for plotting / spatial structure)
                gx = dx * CELLS_PER_DISTRICT_SIDE + cx
                gy = dy * CELLS_PER_DISTRICT_SIDE + cy
                records.append({
                    "district_id": district_id,
                    "gx": gx,
                    "gy": gy,
                })

world = pd.DataFrame(records)
n_cells = len(world)
print(f"Simulated {world['district_id'].nunique()} districts, "
      f"{n_cells} fine cells total "
      f"({n_cells // world['district_id'].nunique()} cells/district)")

# ----------------------------------------------------------------------
# 2. Simulate covariates and a HIDDEN true risk surface
# ----------------------------------------------------------------------

max_gx, max_gy = world["gx"].max(), world["gy"].max()

# Population: denser near several scattered "urban centers"
n_urban = 10
urban_centers = list(zip(rng.uniform(0, max_gx, n_urban), rng.uniform(0, max_gy, n_urban)))
pop = np.zeros(n_cells)
for ux, uy in urban_centers:
    d = np.sqrt((world["gx"] - ux) ** 2 + (world["gy"] - uy) ** 2)
    pop += rng.uniform(2000, 6000) * np.exp(-d / 5)
pop += rng.uniform(50, 300, n_cells)  # rural baseline population
world["population"] = pop

# NDVI-like vegetation index: driven by an INDEPENDENT set of "forest patch"
# centers (deliberately not tied to the urban centers) plus noise, so that
# population and NDVI are not perfectly collinear across districts.
n_forest = 10
forest_centers = list(zip(rng.uniform(0, max_gx, n_forest), rng.uniform(0, max_gy, n_forest)))
ndvi = np.full(n_cells, 0.25)
for fx, fy in forest_centers:
    d = np.sqrt((world["gx"] - fx) ** 2 + (world["gy"] - fy) ** 2)
    ndvi += 0.5 * np.exp(-d / 4)
ndvi += rng.normal(0, 0.05, n_cells)
world["ndvi"] = np.clip(ndvi, 0, 1)

# Rainfall: smooth spatial gradient + noise (e.g. wetter in the south)
rainfall = 50 + 3 * world["gy"] + rng.normal(0, 10, n_cells)
world["rainfall"] = np.clip(rainfall, 0, None)

# ---- Hidden TRUE relative risk (log-linear in covariates) ----
# We pick "true" coefficients that a real disease might plausibly follow:
# risk increases with population density, moderate NDVI (breeding habitat),
# and rainfall (monsoon-linked vector breeding).
true_beta0 = -11.0
true_beta_pop = 0.00035          # more people -> more reported cases
true_beta_ndvi = 2.0             # vegetation proxy for breeding sites
true_beta_rain = 0.02            # rainfall proxy for standing water

log_true_rate = (
    true_beta0
    + true_beta_pop * world["population"]
    + true_beta_ndvi * world["ndvi"]
    + true_beta_rain * world["rainfall"]
)
true_lambda = np.exp(log_true_rate) * world["population"]  # expected cases per cell
world["true_lambda"] = true_lambda
world["true_cases"] = rng.poisson(true_lambda)  # HIDDEN ground truth (never used in fitting)

print(f"\nHidden truth: total cases across all cells = {world['true_cases'].sum()}")

# ----------------------------------------------------------------------
# 3. Aggregate hidden truth to district level -> this is our "observed" data
# ----------------------------------------------------------------------

district_totals = world.groupby("district_id")["true_cases"].sum().rename("observed_cases")
world = world.merge(district_totals, on="district_id")

print(f"Observed district totals (this is ALL we get in real life):")
print(district_totals.describe())

# ----------------------------------------------------------------------
# 4. Fit the disaggregation model
#    Poisson regression where the likelihood is evaluated at the
#    AGGREGATED (district) level, never at the cell level.
# ----------------------------------------------------------------------

# Standardize covariates (excluding the population offset itself, which stays
# as the exposure term) so the optimizer isn't fighting wildly different
# scales -- this matters a lot once we plug in real covariates like
# population (thousands) alongside rainfall (tens) and NDVI (0-1).
raw_cov = np.column_stack([
    world["population"].values,
    world["ndvi"].values,
    world["rainfall"].values,
])
cov_mean = raw_cov.mean(axis=0)
cov_std = raw_cov.std(axis=0)
cov_scaled = (raw_cov - cov_mean) / cov_std

X = np.column_stack([np.ones(n_cells), cov_scaled])
pop_offset = world["population"].values  # exposure term (kept in real units)
district_idx = world["district_id"].values
unique_districts = np.unique(district_idx)
observed = district_totals.loc[unique_districts].values


def neg_log_likelihood(beta):
    """Poisson negative log-likelihood, aggregated to district level."""
    log_rate = np.clip(X @ beta, -30, 30)  # guard against overflow in exp()
    lam_cell = np.exp(log_rate) * pop_offset
    # aggregate predicted lambda up to district level
    lam_district = pd.Series(lam_cell, index=district_idx).groupby(level=0).sum().loc[unique_districts].values
    lam_district = np.clip(lam_district, 1e-10, None)
    ll = np.sum(observed * np.log(lam_district) - lam_district)
    return -ll


beta_init = np.zeros(4)
result = minimize(neg_log_likelihood, beta_init, method="L-BFGS-B",
                   options={"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-10})

print(f"\nOptimization {'succeeded' if result.success else 'DID NOT converge'}")
print(f"Fitted coefficients (standardized-covariate scale): {result.x}")

fitted_beta = result.x

# Convert back to interpretable (original-covariate-scale) coefficients so we
# can compare against the true generating coefficients directly.
fitted_beta_original = fitted_beta[1:] / cov_std
fitted_intercept_original = fitted_beta[0] - np.sum(fitted_beta[1:] * cov_mean / cov_std)
print(f"Fitted coefficients (original scale): "
      f"[{fitted_intercept_original:.4f}, {fitted_beta_original[0]:.6f}, "
      f"{fitted_beta_original[1]:.4f}, {fitted_beta_original[2]:.4f}]")
print(f"True coefficients:                    "
      f"[{true_beta0}, {true_beta_pop}, {true_beta_ndvi}, {true_beta_rain}]")

# ----------------------------------------------------------------------
# 5. Disaggregate: compute cell-level predictions, then RESCALE so that
#    predictions sum exactly to the observed district total (mass-preserving)
# ----------------------------------------------------------------------

log_rate_hat = np.clip(X @ fitted_beta, -30, 30)
lam_cell_hat = np.exp(log_rate_hat) * pop_offset
world["predicted_lambda_raw"] = lam_cell_hat

# rescale within each district so sum(predicted) == observed district total exactly
district_pred_sum = world.groupby("district_id")["predicted_lambda_raw"].transform("sum")
world["predicted_cases"] = (
    world["predicted_lambda_raw"] / district_pred_sum * world["observed_cases"]
)

# ----------------------------------------------------------------------
# 6. Validate against the hidden truth
# ----------------------------------------------------------------------

from scipy.stats import spearmanr, pearsonr

spearman_corr, _ = spearmanr(world["true_cases"], world["predicted_cases"])
pearson_corr, _ = pearsonr(world["true_cases"], world["predicted_cases"])

# Top-decile hotspot overlap: does our method find the same "hottest" cells as truth?
top_frac = 0.1
n_top = max(1, int(top_frac * n_cells))
true_top = set(world.nlargest(n_top, "true_cases").index)
pred_top = set(world.nlargest(n_top, "predicted_cases").index)
overlap = len(true_top & pred_top) / n_top

print(f"\n--- VALIDATION AGAINST HIDDEN TRUTH ---")
print(f"Spearman correlation (true vs predicted fine-cell cases): {spearman_corr:.3f}")
print(f"Pearson correlation:  {pearson_corr:.3f}")
print(f"Top-{int(top_frac*100)}% hotspot cell overlap: {overlap:.1%} "
      f"({len(true_top & pred_top)}/{n_top} cells match)")

# naive baseline for comparison: even split of district total across its cells
world["naive_cases"] = world["observed_cases"] / CELLS_PER_DISTRICT_SIDE ** 2
naive_spearman, _ = spearmanr(world["true_cases"], world["naive_cases"])
naive_top = set(world.nlargest(n_top, "naive_cases").index)
naive_overlap = len(true_top & naive_top) / n_top
print(f"\n--- NAIVE BASELINE (even split across cells) FOR COMPARISON ---")
print(f"Spearman correlation: {naive_spearman:.3f}")
print(f"Top-{int(top_frac*100)}% hotspot overlap: {naive_overlap:.1%}")

# ----------------------------------------------------------------------
# 7. Visualize
# ----------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

grid_true = world.pivot(index="gy", columns="gx", values="true_cases")
grid_pred = world.pivot(index="gy", columns="gx", values="predicted_cases")
grid_naive = world.pivot(index="gy", columns="gx", values="naive_cases")

vmax = max(grid_true.values.max(), grid_pred.values.max())

im0 = axes[0].imshow(grid_true, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
axes[0].set_title("Hidden TRUE fine-cell cases\n(never seen by the model)")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(grid_pred, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
axes[1].set_title(f"Disaggregation model prediction\nSpearman r={spearman_corr:.2f}, "
                   f"hotspot overlap={overlap:.0%}")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(grid_naive, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
axes[2].set_title(f"Naive baseline (even split)\nSpearman r={naive_spearman:.2f}, "
                   f"hotspot overlap={naive_overlap:.0%}")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

for ax in axes:
    # draw district boundaries
    for k in range(1, N_DISTRICTS_X):
        ax.axvline(k * CELLS_PER_DISTRICT_SIDE - 0.5, color="cyan", linewidth=0.8, alpha=0.6)
    for k in range(1, N_DISTRICTS_Y):
        ax.axhline(k * CELLS_PER_DISTRICT_SIDE - 0.5, color="cyan", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("fine-cell x")
    ax.set_ylabel("fine-cell y")

plt.tight_layout()
plt.savefig("/home/claude/disaggregation_validation.png", dpi=150)
print("\nSaved figure to disaggregation_validation.png")
