# PROJECT HANDOVER & CONTEXT RECONSTRUCTION PACKAGE
**VectorHotspot: Dengue-Malaria Dual-Disease Spatiotemporal Early Warning System (India)**

*Prepared for migration to a new Claude account. Reconstructed entirely from conversation history and project memory — nothing in this document is invented. Anything not established in the source conversation is explicitly marked UNKNOWN/NOT CONFIRMED.*

---

## PART 1 — PROJECT IDENTITY

- **Current project name (GitHub repo):** `VectorHotspot` — https://github.com/ThaufeeqAhamed/VectorHotspot (private repo)
- **Previous/informal names:** "Dengue Malaria Research Blueprint" (name of the original uploaded source document); local project folder named `dengue malaria` on the user's Windows machine (`W:\dengue malaria`)
- **Formal research/paper title:** UNKNOWN/NOT CONFIRMED — no formal paper title has been chosen
- **Short description:** A pipeline to forecast dengue and malaria risk in India at fine spatial resolution (H3 hexagons, ~5.2 km² each) despite real surveillance data only being available at coarse resolution (state-level for dengue, district-level for malaria), using statistical disaggregation to bridge the gap, with planned downstream forecasting and hotspot detection.
- **One-paragraph explanation:** India's public dengue and malaria surveillance data is only available at state or district resolution — far too coarse for genuinely local early-warning. This project uses real environmental/demographic covariates (currently population; more planned) to statistically disaggregate coarse case counts down to H3 hexagon resolution in a mass-preserving way (hexagon-level predictions always sum back to the real known state/district total). The disaggregated fine-resolution case surface is intended to feed a forecasting pipeline (LightGBM/XGBoost per the original blueprint) and hotspot detection (Getis-Ord Gi* on *predicted* future risk, not historical data), validated against subsequently-observed real hotspots.
- **Detailed explanation:** See PART 6 (methodology) and PART 2 (evolution) below.
- **Core problem being solved:** Reconciling the mismatch between coarse real-world disease surveillance resolution and the fine spatial resolution needed for actionable local early warning, without resorting to statistically indefensible naive disaggregation (even-splitting).
- **Target users/stakeholders:** UNKNOWN/NOT CONFIRMED in detail — implied academic audience (faculty reviewing the GitHub repo). No formal stakeholder analysis (e.g., specific health department, NGO) was discussed.
- **Real-world motivation:** Dengue and malaria are major vector-borne disease burdens in India; earlier/finer-grained warning could in principle help resource allocation. Not elaborated further by the user.
- **Academic motivation:** This is a student project being submitted to a faculty member for review via a private GitHub repository.
- **Expected final output:** A GitHub repository containing the full reproducible pipeline (data collection → cleaning → spatial disaggregation → [planned: temporal disaggregation → feature engineering → forecasting → hotspot detection → validation → dual-disease comparison → explainability → dashboard]), plus documentation. Whether a formal written research paper is also required: UNKNOWN/NOT CONFIRMED.
- **Research objective (reconstructed, not literally user-stated as a formal RQ):** Can dengue and malaria case counts, available in India only at state/district resolution, be validly disaggregated to H3-hexagon resolution using open covariates, and can the resulting fine-resolution risk surface support meaningful future-hotspot prediction validated against real subsequently-observed outcomes?
- **Engineering objective:** Build a fully reproducible pipeline using only real (non-synthetic) data in the final version; synthetic data was explicitly used ONLY to prove the disaggregation method works before touching real data, per user's explicit instruction ("we dont want... synthetic ones").
- **SDG alignment:** UNKNOWN/NOT CONFIRMED — never discussed by the user, though the subject matter self-evidently relates to public health.
- **Current maturity/stage:** Mid-pipeline. All data collection (Tier 1 mandatory datasets) is complete and validated. The H3 spatial grid is complete and validated. The spatial disaggregation model is complete, validated, and has produced real, epidemiologically-sensible results. Everything downstream of spatial disaggregation (temporal disaggregation, feature engineering, forecasting models, hotspot detection, validation, explainability, dashboard, Tier 2 covariates) has NOT been started.

---

## PART 2 — ORIGINAL IDEA AND EVOLUTION

### Chronological evolution

```
ORIGINAL IDEA (from uploaded "Dengue_Malaria_Research_Blueprint.docx")
  Full pipeline: multi-source data -> H3 grid (as the NATIVE prediction unit)
  -> multi-factor feature engine -> separate LightGBM/XGBoost models per disease
  -> 1-4 week risk forecast -> Getis-Ord Gi* on PREDICTED risk surface
  -> future hotspot detection -> validation against subsequently-observed hotspots
  -> dual-disease comparison -> SHAP explainability + calibration
  -> early warning engine -> dashboard
  Plus a separate "Research Evaluation" layer: baselines, ablations,
  H3-vs-admin-boundary comparison, spatial/temporal holdout validation,
  hotspot method comparison, disease comparison.
        |
INITIAL REVIEW (by Claude, before any data work)
  Identified that the blueprint ITSELF already correctly disclaimed novelty
  for individual components (LightGBM/XGBoost/SHAP/Gi*/climate features are
  not novel on their own) -- the real contribution is the COMBINATION,
  especially forecast-to-hotspot fusion and rigorous future-hotspot validation.
  Flagged open gaps: malaria data source unresolved, H3 resolution deferred
  but actually dictated by case data's native resolution, target variable
  formulation (regression vs classification) undecided, dengue/malaria
  likely need different lag windows (different vector biology).
        |
DATASET DISCOVERY PHASE (extensive real-world verification)
  Systematically checked real access to every planned data source. Found:
  - NCVBDC public dengue page: only STATE-level, ANNUAL (not district/weekly)
  - NCVBDC public malaria page (MMIS): only NATIONAL/risk-category level
  - IHIP (India's real fine-resolution VBD portal): LOGIN-GATED, not public
  - OpenDengue's actual India data (verified by loading the real file):
    STATE-level (Admin1) only, ANNUAL, 2010-2024, zero district-level rows
  - EpiClim (an initially promising Zenodo dataset): turned out to be sparse
    OUTBREAK-REPORT data (~9,000 rows for all India/diseases/15 years), not
    a continuous weekly series -- downgraded to secondary/not used further
  - ONE major exception found: a real NCVBDC PDF ("District wise Malaria
    Data 2000 to 2024") that DOES have genuine district-level annual data
    for malaria specifically -- this became the malaria data source
        |
USER'S CRITICAL QUESTION: "the datasets available online are of district or
state level not of minute level... i think i cant apply [H3] to this
project... is this still possible to build using our approach??"
        |
MAJOR PIVOT: reframe from "H3 as native prediction resolution" to
"statistical disaggregation" (in the style of the Malaria Atlas Project /
WorldPop's model-based geostatistical disaggregation approach)
  REASON: the resolution mismatch that sinks most similar projects IS
  ITSELF treated as the research gap -- most groups either quietly accept
  coarse resolution or fake fine resolution via naive even-splitting
  (statistically indefensible). Proper covariate-informed disaggregation,
  validated against finer ground truth where available, is the actual
  novel contribution.
  NEW ARCHITECTURE: insert a disaggregation stage BEFORE the original
  feature engine: [coarse case counts + fine covariates] ->
  [mass-preserving statistical disaggregation model] -> [H3-cell case
  surface] -> [original blueprint's feature engine -> forecast -> Gi* ->
  validation pipeline, unchanged]
        |
METHOD PROVEN ON SYNTHETIC DATA FIRST (explicit user instruction: prove the
method before touching real data, and use NO synthetic data in the final
version)
  Built disaggregation_prototype.py: Poisson regression with a
  population-like covariate + aggregation constraint, mass-preserving
  rescale. First naive version (25 districts, correlated covariates,
  Nelder-Mead) gave WEAK results (55.6% hotspot overlap vs 51.1% naive
  baseline) due to identifiability problems. Fixed version (225 districts,
  DEcorrelated covariates, L-BFGS-B, standardized covariates) gave STRONG
  results: 91.9% Spearman correlation vs 89.8% for naive even-split;
  86.7% top-10% hotspot overlap vs 74.6% for naive. This was the go-ahead
  to proceed to real data.
        |
REAL DATASET COLLECTION (Tier 1 mandatory, see PART 9 for full inventory)
  All 6 datasets collected, cleaned, and validated: dengue (state/annual),
  malaria (district/annual, required fixing a major PDF-parsing bug --
  see PART 13), rainfall+temperature (district/weekly, required extensive
  debugging -- see PART 13), population (district AND later hexagon level,
  WorldPop), administrative boundaries (724 districts, cleaned).
        |
USER'S SECOND CRITICAL PUSH: after H3 grid resolution was first proposed at
resolution 6 (~91,000 hexagons, ~36 km2/cell, "tehsil-sized"), user
explicitly rejected this: "we dont want like how everybody others are doing
like making district wise prediction we are supposed to make predictions
for minute areas too."
        |
RESOLUTION REVISED TO 7 (~620,742 hexagons actually generated, ~5.2 km2/cell,
"village-cluster-sized," ~877x finer than district-level) -- explicitly
justified as being close to the practical ceiling of what real covariates
(1km population, ~25km rainfall grid, ~100km temperature grid) can actually
support without fabricating false precision; resolution 8 was considered
and rejected as adding row count without adding genuine information given
current covariates.
        |
H3 GRID GENERATED, VALIDATED (620,742 real hexagons, all 724 districts/36
states covered, visually confirmed correct India shape)
        |
PER-HEXAGON POPULATION COMPUTED (the fine covariate the disaggregation model
actually needs -- district-level population alone can't explain WITHIN-
district spatial variation, which is the whole point of disaggregation)
        |
REAL DISAGGREGATION MODEL WIRED AND FIT on real data (dengue state->hex,
malaria district->hex), using population as the covariate. Weather was
DELIBERATELY excluded from this spatial disaggregation step (varies too
little within most districts to explain spatial redistribution) and
reserved for the later TEMPORAL forecasting stage instead.
        |
MAJOR DATA QUALITY BUG DISCOVERED AND FIXED during this step: 52% of
malaria rows had invalid (state,district) pairs, traced to the original
PDF parsing silently mislabeling district blocks under the wrong state
(see PART 13 for full detail). Fixed down to a 14.2% residual (explained,
not a bug -- see PART 13).
        |
CURRENT STATE: real disaggregation model complete, validated, results are
epidemiologically sensible (dengue positive population coefficient/urban-
concentrated, malaria negative coefficient/rural-forest-concentrated --
visually confirmed against real known geography). This is the LATEST
completed phase. Awaiting decision: pull Tier 2 covariates next, or move
to temporal disaggregation / feature engineering / forecasting.
```

### Ideas considered but abandoned
- **Naive even-split disaggregation** (splitting a district's case count evenly across its hexagons/area) — explicitly proven inferior in the synthetic prototype test (74.6% vs 86.7% hotspot overlap) and rejected as "statistically indefensible."
- **H3 resolution 6** as the final target — rejected per the user's explicit push for genuinely fine ("minute area") resolution, not "district-wise dressed up as hexagons."
- **Staying at district-level prediction** (i.e., abandoning H3 entirely after discovering the real-data resolution mismatch) — considered by the user as a worry ("i think i cant apply to this project"), explicitly rejected in favor of the disaggregation pivot.

### Ideas considered but postponed (not abandoned)
- **Tier 2 covariates** (NDVI, land cover, water bodies) — identified as "strongly recommended," sources verified as accessible (MODIS via NASA Earthdata, ESA WorldCover, JRC Global Surface Water), but NOT YET PULLED. This is the immediate open decision point as of this handover.
- **Temporal disaggregation** (redistributing annual case totals across weeks using weekly weather as a guide) — identified as necessary since case data is annual-only but the blueprint wants weekly forecasts. NOT YET BUILT.
- **Human mobility, intervention data (vector control/spraying), entomological/vector surveillance data** — explicitly marked "optional/future work" in the ORIGINAL blueprint document itself. NOT pursued yet.
- **Disease co-occurrence features** (using one disease's activity as a covariate for the other) — marked "optional but interesting, should be tested, not mandatory" in the original blueprint. NOT implemented yet.
- **City ward-level dengue validation** (Bhopal Municipal Corporation ward-level GeoJSON, 86 wards with population, separately found/uploaded by the user) — collected and saved as a Tier 3 "validation-critical" asset, but NOT YET USED to actually validate the disaggregation model's within-district spatial pattern. This is a genuine open task, not something rejected.

### Current final direction
Real-data-only pipeline; H3 resolution 7; statistical disaggregation (not naive splitting, not staying coarse) as the core spatial methodology; population as the current sole real covariate; weather reserved for temporal/forecasting stage; original blueprint's forecast->Gi*->validation pipeline still the intended downstream architecture, not yet built.

---

## PART 3 — RESEARCH PROBLEM

- **Problem statement:** Real Indian vector-borne disease surveillance data is only publicly available at coarse (state or district) resolution, which prevents genuinely fine-grained (H3-hexagon-level) disease risk prediction and hotspot early-warning.
- **Existing real-world problem:** Public health resource allocation decisions currently can't be informed by fine-grained spatial risk prediction because the underlying case data doesn't support it directly.
- **Existing technical limitations (ESTABLISHED FACT, verified firsthand during this project):**
  - NCVBDC's public dengue page: state-level, annual only
  - NCVBDC's public malaria page (MMIS): national/risk-category level only
  - IHIP (India's actual fine-resolution VBD reporting system): login-gated, not publicly accessible
  - OpenDengue's India extract: state-level (Admin1) only, despite the source paper claiming improved South Asian subnational disaggregation generally
  - City ward-level data exists only as numbers scattered across news articles/press releases, not as structured downloadable datasets
- **Existing research limitations (RESEARCH INTERPRETATION, not independently verified against literature — no formal lit review was done):** Most similar disease-mapping research either (a) quietly accepts coarse resolution, or (b) fakes fine resolution via naive even-splitting, which is statistically indefensible. This claim is based on the project's own reasoning during the pivot discussion, NOT on a verified literature survey.
- **Identified research gap (OUR PROPOSED CONTRIBUTION):** Validated statistical disaggregation of dual-disease (dengue AND malaria jointly) coarse surveillance data to H3 resolution using real, verifiable multi-source covariates, combined with forecast-to-hotspot fusion and rigorous future-hotspot validation.
- **Proposed contribution (as currently articulated in this project):**
  1. Validated statistical disaggregation of coarse dual-disease data to H3 resolution using multi-source covariates (headline contribution)
  2. H3-native multi-factor forecasting for both diseases (PLANNED, not built)
  3. Forecast-to-hotspot fusion: Gi* run on PREDICTED risk, not historical case data (PLANNED, not built)
  4. Rigorous future-hotspot validation: predicted hotspot vs. subsequently-observed hotspot (PLANNED, not built)
  5. Dual-disease comparative analysis (PLANNED, not built)
- **Why the proposed approach addresses the gap:** By using real fine-resolution covariates (population, and planned NDVI/land cover/water) to inform WHERE WITHIN a coarse unit cases likely concentrate, rather than assuming uniform distribution, the disaggregation is statistically defensible and — where finer ground truth exists (e.g., Bhopal ward data) — checkable.
- **What makes this different from a basic ML prediction project:** The disaggregation VALIDATION step (checking disaggregated patterns against held-out finer data) and the FUTURE-hotspot validation step (checking predictions against what subsequently actually happened, not just historical fit) are the rigor-adding pieces most comparable projects skip. NOTE: the disaggregation validation against Bhopal ward data specifically has NOT yet been executed — it remains an open task (see PART 28).
- **Novelty caveat (explicitly established early in the project, still valid):** The project does NOT claim novelty for LightGBM, XGBoost, SHAP, Getis-Ord Gi*, or climate features individually — these are standard, well-established techniques. The claimed novelty is specifically in the combination and the disaggregation-validation methodology.

---

## PART 4 — RESEARCH QUESTIONS AND OBJECTIVES

**IMPORTANT CAVEAT:** No formal, explicitly-worded research question, hypothesis, or success-criteria document was ever produced or requested by the user during this project. Everything below is RECONSTRUCTED from the project's actual direction and decisions, and should be treated as RESEARCH INTERPRETATION, not verbatim user statements, unless marked otherwise.

- **Main research question (reconstructed):** Can dengue and malaria case counts, available in India only at state/district resolution, be validly disaggregated to H3-hexagon resolution using open covariates, and can the resulting fine-resolution risk surface support meaningful future-hotspot prediction validated against real subsequently-observed outcomes?
- **Secondary questions (reconstructed):**
  - Does H3-based spatial-neighbor structure outperform naive administrative-adjacency for feature engineering, holding the prediction target fixed? (Flagged as ablation E3 in the original blueprint, made MORE central after the disaggregation pivot. NOT YET RUN.)
  - Do dengue and malaria exhibit different population-risk relationships? **ANSWERED empirically, real result:** yes — dengue's fitted population coefficient is positive (risk increases faster than population, consistent with urban/dense-area concentration), malaria's is negative (risk increases as population decreases, consistent with rural/forest-fringe concentration). See PART 16 for the actual fitted values available in `disaggregation_fit_report.txt`.
- **Objectives:** See the phase-by-phase roadmap in PART 25.
- **Hypotheses:** None were formally pre-registered. The dengue-urban / malaria-rural finding could retrospectively be framed as a confirmed hypothesis, but it was not stated as a hypothesis in advance — it emerged from fitting real data.
- **Success criteria:** Not formally defined by the user. Implicitly: a working, reproducible, real-data-only pipeline that survives faculty review, with the original blueprint's planned rigorous validation (precision/recall/F1/spatial IoU/lead time for future-hotspot detection) as the evidentiary bar — though this specific validation has not yet been re-confirmed as still the plan after the disaggregation pivot.

---

## PART 5 — LITERATURE REVIEW AND RESEARCH KNOWLEDGE

**No formal literature review was conducted in this project.** This section documents the only research/methodology references that came up, with explicit honesty about their verification status.

- **Malaria Atlas Project (MAP), Oxford** — mentioned as the real-world precedent for the disaggregation methodology being used (their 1km-resolution malaria risk maps are produced via model-based geostatistical disaggregation from aggregate case data). NOT independently verified against a specific cited paper — mentioned by name/reputation only, no title/authors/year captured.
- **WorldPop's `disaggregation` R package** — mentioned as an existing open-source implementation of the same general class of method (Poisson/binomial regression with an aggregation constraint). NOT used directly in this project (a custom Python implementation via `scipy.optimize` was built instead) — mentioned only as methodological precedent/inspiration.
- **OpenDengue project** — its own paper (not formally cited with title/author) was referenced once regarding a claim of "greater subnational disaggregation of data from South Asia" in a v1.2 update — this claim was CHECKED AGAINST THE ACTUAL DATA and found NOT to hold for India specifically (the real downloaded file has zero Admin2/district rows for India, only Admin1/state). This is a verified empirical finding, not a literature claim taken at face value.
- **No other papers, authors, or specific citations were discussed, searched for, or verified anywhere in this project.**
- **Literature review conclusions:** NOT APPLICABLE — no review was performed.
- **Identified research gaps:** See PART 3 (research-gap framing came from the project's own data-access investigation, not from a literature survey).
- **Methodological lessons:** The core lesson driving the disaggregation approach was empirical (real data access investigation), not literature-derived.
- **Approaches decided not to use:** See PART 2 ("ideas considered but abandoned") and PART 15.
- **Important research terminology established in this project:**
  - "Mass-preserving disaggregation" — hexagon-level predictions rescaled so they sum exactly to the known coarse (state/district) total
  - "Tier 1 / Tier 2 / Tier 3" datasets — mandatory / strongly-recommended / validation-critical, a prioritization scheme established early in dataset planning
  - "Forecast-to-hotspot fusion" — running Gi* hotspot detection on PREDICTED future risk rather than historical case data (from the original blueprint, a stated novelty point)

---

## PART 6 — CURRENT PROPOSED METHODOLOGY (FULL PIPELINE)

This reflects the CURRENT state: original blueprint stages are listed, with a note on what's actually been built vs. still planned.

| Stage | Input | Output | Method/Tool | Status |
|---|---|---|---|---|
| Data collection | Public sources (NCVBDC, OpenDengue, IMD, WorldPop, community GeoJSON boundaries) | 6 Tier 1 raw/cleaned datasets | Web search, PDF parsing (`pdfplumber`), manual download+upload (large files couldn't be auto-fetched) | COMPLETE |
| Data cleaning | Raw datasets | Validated CSVs | pandas, extensive manual bug-fixing (see PART 13) | COMPLETE for Tier 1 |
| Spatial processing (boundaries) | Community GeoJSON (760 raw features) | 724 clean districts, all India | Dedup, (state,district)-keyed validation | COMPLETE |
| Spatial processing (H3 grid) | 724 district polygons | 620,742 resolution-7 hexagons, tagged with district+state | `h3-py` v4, `geopandas`, point-in-polygon spatial join | COMPLETE |
| Spatial disaggregation | State-level dengue + district-level malaria annual totals + hexagon population | Hexagon-level annual case estimates, mass-preserving | Custom Poisson regression (`scipy.optimize`, L-BFGS-B), fit once globally per disease, aggregated log-likelihood | COMPLETE |
| Temporal disaggregation | Hexagon-level ANNUAL case estimates + weekly weather | Hexagon-level WEEKLY case estimates | NOT DESIGNED YET — only identified as necessary | PLANNED, not started |
| Feature engineering | Weekly hex-level data | Lag/rolling/growth-rate/neighbor/seasonality features | H3 k-ring neighbor queries, pandas rolling windows (conceptually planned per original blueprint) | PLANNED, not started |
| Disease forecasting models | Engineered features | 1-4 week-ahead risk prediction, per disease | LightGBM/XGBoost (per original blueprint) | PLANNED, not started |
| Hotspot detection | Predicted risk surface | Predicted future hotspots | Getis-Ord Gi* on PREDICTED (not historical) risk | PLANNED, not started |
| Future hotspot validation | Predicted hotspots + subsequently-observed real data | Precision/Recall/F1/Spatial IoU/Lead time | Not yet designed in detail | PLANNED, not started |
| Dual-disease comparison | Both diseases' hotspot outputs | Overlap/comparison analysis | Not yet designed | PLANNED, not started |
| Explainability | Trained forecast models | SHAP per-cell/per-hotspot explanations | SHAP (per original blueprint) | PLANNED, not started |
| Uncertainty/calibration | Forecast models | Confidence/prediction intervals | Marked "Partial" in ORIGINAL blueprint's own self-assessment; not started in actual implementation | PLANNED, not started |
| Early warning engine | Risk + Gi* significance + confidence + lead time | Combined alert output | Per original blueprint | PLANNED, not started |
| Dashboard | All of the above | User-facing visualization | Per original blueprint | PLANNED, not started |
| Research evaluation | All of the above | Baselines, ablations, holdouts | Per original blueprint's own "Research Evaluation" box | PLANNED, not started |

---

## PART 7 — SPATIAL METHODOLOGY (DETAILED)

- **Geographic units:** 724 districts, 36 states/UTs (cleaned from a raw 760-feature community GeoJSON — 34 duplicate whole-state-outline features and 2 exact-duplicate features for Chandigarh/Lakshadweep were removed). Source: `udit-001/india-maps-data` GitHub repo (community-maintained, includes post-2019 district splits e.g. Mizoram's Khawzawl/Hnahthial).
- **Administrative boundaries known data-quality caveats (documented, not fixed — accepted as known limitations):**
  - `dt_code` is NOT a reliable unique identifier (only unique within a state, not nationally; Tamil Nadu's newest districts use placeholder code `'0'`; Gujarat has a genuine code collision between two real different districts). **DECISION: always join on (state name, district name), never on `dt_code` alone.**
  - The boundary file reflects India's OFFICIALLY CLAIMED territory, not actual administrative control — e.g., Ladakh's mapped area is far larger than commonly-cited figures because it includes India-claimed-but-not-controlled Aksai Chin; similarly a "Mirpur" district exists in the data that is actually in Pakistan-administered Kashmir. This causes some population/case-matching discrepancies for those specific units, documented as known limitations, not bugs.
- **H3 library:** `h3-py` v4 API (`h3.LatLngPoly`, `h3.polygon_to_cells`, `h3.cell_to_latlng`, `h3.latlng_to_cell`).
- **H3 resolution: 7** (final decision — see PART 2 and PART 14 for the reasoning and the resolution-6-to-7 revision).
  - Average cell size ~5.2 km² (~2.3 km across)
  - **Actual generated count: 620,742 hexagons** (not the initially-estimated ~635,000 — close, difference is normal/expected from real coastline/boundary complexity)
- **H3 grid construction method:** Union all 724 district polygons into a single India boundary FIRST (via `geopandas` `union_all()`/`unary_union`), then generate the hex grid over the union in one pass (avoids duplicate/ambiguous cells at district borders that per-district generation would cause). Each hexagon is then assigned to a district via point-in-polygon spatial join on the hexagon's CENTER point (`geopandas.sjoin`, `predicate="within"`).
- **Edge case handling:**
  - Coastal hexagons whose center fell just outside every polygon (0 occurred in the final validated run): nearest-district-centroid fallback logic exists but was not needed.
  - Districts smaller than one hexagon (Lakshadweep, Puducherry's Mahe exclave — 2 districts): force-assigned one hexagon snapped to their centroid.
  - **Bug found and fixed:** initial validation code checked district coverage using district NAME alone, which silently masked genuinely-missing districts when names repeated across states (5 such name collisions exist nationally: Aurangabad, Balrampur, Bilaspur, Hamirpur, Pratapgarh). Fixed by keying all coverage/validation logic on (state, district) tuples. Final validated result: **724/724 (state,district) pairs covered, 36/36 states covered, 0 duplicate hexagon IDs.**
- **Population allocation (per-hexagon, the key disaggregation covariate):**
  - Source: WorldPop, 1km resolution, 5 anchor years (2000, 2005, 2010, 2015, 2020) — the 100m product exists but is far larger (1.7GB/year vs ~18MB/year at 1km) and was judged unnecessary precision given population doesn't vary meaningfully week-to-week for this use case.
  - Method (final, fast version): bulk `rasterio.features.rasterize()` of all 620,742 hexagon polygons at once against each year's raster grid, `all_touched=True`, then `np.bincount`-based summation — NOT per-feature `zonal_stats()` calls, which were tested and found to take 50+ minutes PER YEAR at this hexagon count (would have been 4+ hours total).
  - **Known, accepted limitation of this fast method:** since it assigns each raster pixel to exactly one hexagon ("winner takes all" for pixels straddling a hexagon boundary), there's a small systematic bias at hexagon edges. Cross-validated as acceptable: national population totals matched known district totals within 0.15%, 75% of districts within 1.2% error. The few larger-error districts (Mahe, Diu, Mirpur, Nicobars — all under 2% of districts) are explained by genuine geometric/political edge cases, not computation errors.
  - **Population years beyond the WorldPop range:** for case-data years outside 2000-2020, population is interpolated LINEARLY between the nearest two anchor years, or held FLAT at the 2020 value for years after 2020 (no newer WorldPop data available). This is a stated simplifying assumption, not a bug.
- **Mass-preserving disaggregation — exact method:**
  - **Input:** coarse case totals (state-annual for dengue, district-annual for malaria) + hexagon population (interpolated per year, per above)
  - **Model:** `E[cases_hex] = population_hex * exp(beta0 + beta1 * standardized_log1p(population_hex))` — population enters BOTH as a multiplicative exposure offset (cases scale with population, all else equal) AND as a covariate (letting the model learn whether risk is super- or sub-linear in population density).
  - **Fitting:** ONE global Poisson regression per disease, fit by maximizing the Poisson log-likelihood evaluated at the AGGREGATED (state or district) level across ALL (unit, year) combinations simultaneously — the model NEVER sees hexagon-level case counts (they don't exist), only real unit-level totals. Optimized via `scipy.optimize.minimize`, method `L-BFGS-B`. Implemented with an efficient vectorized grouping (`pandas.groupby().ngroup()` + `np.bincount`), NOT a Python-tuple-based grouping (which caused an out-of-memory crash during testing — see PART 13).
  - **Conservation constraint (mass-preservation):** after fitting, hexagon-level predicted rates are rescaled within each (unit, year) group so they sum EXACTLY to that unit's real observed case total: `predicted_cases_hex = raw_prediction_hex / sum(raw_predictions in unit) * observed_unit_total`.
  - **Validation performed:** recomputing unit totals from hexagon predictions and comparing to observed — **result: max |predicted_total - observed| = 0.000000 across all 451 dengue unit-years and all 14,402 malaria unit-years** (i.e., exact, by construction, confirmed numerically).
  - **Current implementation status:** COMPLETE for the spatial dimension (annual, population-only covariate). NOT YET extended to include Tier 2 covariates (NDVI/land cover/water) or to the temporal (weekly) dimension.
  - **Known limitation (stated explicitly, not hidden):** population is currently the ONLY real covariate in this model. This is a legitimate first working version, not the final intended model.
- **Spatial disaggregation validated against held-out finer ground truth?** NOT YET DONE. The Bhopal ward-level GeoJSON (86 wards, real population, collected earlier as a Tier 3 validation asset) has NOT been used to check whether the disaggregation model's within-district pattern matches real intra-district variation. This is an explicit open task (see PART 28).
- **Hotspot methodology (Getis-Ord Gi*):** PLANNED per original blueprint, NOT IMPLEMENTED yet in any form.
- **Spatial statistical/validation methods beyond the above:** none implemented yet.

---

## PART 8 — TEMPORAL METHODOLOGY

- **Time granularity of REAL case data:** ANNUAL ONLY for both diseases (dengue: OpenDengue, 2010-2024; malaria: NCVBDC PDF-derived, 2000-2024). This is a hard constraint discovered during data collection — the original blueprint's "H3 cell x week" vision cannot be directly satisfied by real case data at any resolution.
- **Time granularity of weather data:** WEEKLY, district-level, 2000-2024 (IMD rainfall/temperature, aggregated from daily grids).
- **Time granularity of population data:** 5 discrete annual snapshots (2000, 2005, 2010, 2015, 2020), interpolated/extrapolated as described in PART 7.
- **Implication (explicitly identified, not yet resolved):** because case data is annual-only, genuine WEEKLY forecasting (as the original blueprint envisions) requires an additional TEMPORAL disaggregation step — redistributing each hexagon's annual case estimate across ~52 weeks using weekly weather (and possibly other seasonal signals) as a guide, analogous in spirit to the spatial disaggregation already built but along the time axis instead. **This step has NOT been designed or built yet** — it is the most immediately next planned piece of new methodology, alongside or after Tier 2 covariates.
- **Lag features, rolling windows, seasonality features:** PLANNED per the original blueprint (dengue/malaria case lags, rolling means, growth rates, weather lags/rolling stats, H3-neighbor lagged features, sin/cos week-of-year encoding) — NONE implemented yet, since feature engineering hasn't started.
- **Forecast horizon:** 1-4 weeks, per the original blueprint. Not yet implemented, so not yet tested.
- **Train/validation/test temporal split:** NOT YET DEFINED. The original blueprint calls for "rolling temporal + spatial hold-out" validation as part of its Research Evaluation layer — this has not been designed for the real pipeline yet.
- **Leakage prevention:** NOT YET IMPLEMENTED in code. The original blueprint explicitly flagged this as critical ("only information available before the prediction date can be used" for neighbor/lag features) — this principle is DOCUMENTED as a requirement but no leakage-safety code exists yet since feature engineering hasn't started.
- **Why the current (annual, pre-temporal-disaggregation) design exists:** it's a direct consequence of real data availability, not a deliberate simplification of the original weekly vision — the intent remains to reach weekly resolution via temporal disaggregation.

---

## PART 9 — DATASETS (COMPLETE INVENTORY)

| Dataset | Source | Purpose | Granularity | Spatial Level | Temporal Coverage | Status | File/Location |
|---|---|---|---|---|---|---|---|
| Dengue cases | OpenDengue (Spatial_extract, India) | Case counts for disaggregation target | Annual | State (Admin1) | 2010-2024 | Done, in use | `data/processed/dengue_state_2010_2024.csv` |
| Malaria cases | NCVBDC PDF ("District wise Malaria Data 2000 to 2024") | Case counts for disaggregation target | Annual | District | 2000-2024 | Done, in use — required a major post-hoc state-mislabeling fix (see PART 13) | `data/processed/malaria_district_2000_2024.csv` |
| Rainfall + Temperature | India Meteorological Department (IMD), via `imdlib` | Planned covariate for TEMPORAL disaggregation/forecasting (not yet used) | Weekly (aggregated from daily) | District | 2000-2024 | Done, validated — NOT YET consumed by any downstream model | `data/processed/imd_district_weekly_weather_2000_2024_FIXED.csv` |
| Population (district) | WorldPop, 1km | Cross-validation reference for hex population | Annual snapshots | District | 2000, 2005, 2010, 2015, 2020 | Done | `data/processed/district_population_2000_2020.csv` |
| Population (hexagon) | WorldPop, 1km | PRIMARY covariate for spatial disaggregation | Annual snapshots (interpolated) | H3 hexagon | 2000, 2005, 2010, 2015, 2020 | Done, in use | `data/processed/hex_population_2000_2020.csv` |
| Administrative boundaries | `udit-001/india-maps-data` (GitHub, community GeoJSON) | District polygons for H3 grid generation, all spatial joins | Static | District | Current (as of source repo) | Done, cleaned | `data/boundaries/india_districts_clean.geojson` |
| H3 grid | Generated (not downloaded) | Base spatial unit for everything | Static | Hexagon (res 7) | N/A | Done | `data/processed/india_h3_grid_res7.csv` |
| Disaggregated dengue (hex-annual) | Derived (this project's model output) | Fine-resolution dengue case estimates | Annual | Hexagon | 2010-2024 | Done — 409MB, too large for repo, regenerable via script | `data/processed/dengue_hex_annual.csv` (gitignored) |
| Disaggregated malaria (hex-annual) | Derived (this project's model output) | Fine-resolution malaria case estimates | Annual | Hexagon | 2000-2024 | Done — 748MB, too large for repo, regenerable via script | `data/processed/malaria_hex_annual.csv` (gitignored) |
| Bhopal ward boundaries | User-sourced (Bhopal Municipal Corporation, uploaded directly by user) | Tier 3 — intended for validating disaggregation against real finer-than-district ground truth | Static | Ward (86 wards) | Current | Collected, NOT YET USED for validation | `bhopal_wards.geojson` (exact current repo placement not confirmed) |
| EpiClim (Zenodo) | Zenodo record 14580510 | Initially considered as a case-data source | Nominally weekly, actually sparse outbreak-report | District | 2009-2023 | Evaluated and REJECTED as primary source — not used further | Not incorporated into the repo |
| NDVI | MODIS (NASA Earthdata) — Tier 2 | Planned covariate to strengthen disaggregation | 250m, 16-day composite | Raster | TBD | PLANNED, source verified accessible, NOT pulled | N/A |
| Land Surface Temperature | MODIS (NASA Earthdata) — Tier 2 | Planned covariate | 1km, 8-day composite | Raster | TBD | PLANNED, NOT pulled | N/A |
| Land cover | ESA WorldCover 2021 — Tier 2 | Planned covariate | 10m | Raster | 2021 snapshot | PLANNED, source verified accessible, NOT pulled | N/A |
| Water bodies | JRC Global Surface Water — Tier 2 | Planned covariate (esp. relevant to malaria vector ecology) | 30m | Raster | 1984-2021 | PLANNED, source verified accessible, NOT pulled | N/A |
| Entomological/vector surveillance | NCVBDC — Tier 3 | Planned covariate AND validation source | Unknown | Unknown | Unknown | IDENTIFIED as valuable, NOT pursued | N/A |
| Human mobility, intervention data | Not sourced | Explicitly deferred per original blueprint | N/A | N/A | N/A | Explicitly out of scope for "Version 1" per the original blueprint document | N/A |

**Known data-quality issues per dataset (see PART 13 for full debugging narratives):**
- Malaria: originally 52% of rows had invalid (state,district) pairs due to a PDF-parsing bug; fixed to a 14.2% residual (explained, not corrupted).
- Weather: temperature data was originally 66-72% null due to a coarse-grid zonal-stats bug; fixed to 0.4% null (remaining nulls are explained: remote islands, one remote border district).
- Population (hexagon-level): small systematic edge bias from the fast bulk-rasterize method, cross-validated as acceptable.
- Boundaries: `dt_code` unreliable, some districts reflect disputed/claimed territory not actual administrative control.

---

## PART 10 — CODEBASE

```
VectorHotspot/                          (GitHub: ThaufeeqAhamed/VectorHotspot, private)
|-- .gitignore                          (excludes raw rasters, imd_raw_data/, *_hex_annual.csv)
|-- data/
|   |-- boundaries/
|   |   `-- india_districts_clean.geojson
|   |-- raw/
|   |   `-- population/                 (5 WorldPop .tif files -- user must place here to rerun population script)
|   `-- processed/                      (all cleaned/derived CSVs -- see PART 9 table)
|-- imd_raw_data/                       (kept at project ROOT, not under data/ -- large raw weather grids, gitignored)
|   |-- rain/
|   |-- tmax/
|   `-- tmin/
|-- src/
|   |-- data_prep/
|   |   |-- fetch_imd_weather.py        (downloads+aggregates IMD weather; resumable, retry logic)
|   |   |-- recompute_temperature_fast.py  (fixes the temperature zonal-stats bug; fast rasterize+bincount method)
|   |   |-- compute_district_population.py (district-level population from WorldPop rasters)
|   |   |-- generate_h3_grid.py         (generates the 620,742-hexagon grid)
|   |   `-- generate_hex_population.py  (fast per-hexagon population via bulk rasterize+bincount)
|   `-- disaggregation/
|       |-- disaggregation_prototype.py (SYNTHETIC-data proof of concept -- proves the method, not for production use)
|       `-- wire_disaggregation_model.py (the REAL disaggregation model -- dengue+malaria, real data)
|-- outputs/
|   `-- figures/                        (proposed location for .png visualizations -- not confirmed populated yet)
`-- docs/
    `-- brain.md                        (running plain-text project log, predates this formal handover document)
```

**Important files — purpose/inputs/outputs/status:**

- **`generate_h3_grid.py`** — Input: `india_districts_clean.geojson`. Output: `india_h3_grid_res7.csv`. Status: working, validated. Known past bug (fixed): district-coverage validation keyed on district name alone instead of (state,district) — corrected.
- **`generate_hex_population.py`** — Input: H3 grid + 5 WorldPop `.tif` files. Output: `hex_population_2000_2020.csv`. Status: working, validated. Past bug (fixed): initial version used `rasterstats.zonal_stats()` per-hexagon, which took 50+ min/year; rewritten to bulk `rasterize()` + `np.bincount`.
- **`fetch_imd_weather.py`** — Input: none (downloads from IMD via `imdlib`). Output: `imd_district_weekly_weather_2000_2024.csv`. Status: working after extensive fixes (resume/retry logic for IMD's flaky server, path-resolution fixes, a leftover-code duplicate-download bug). NOT YET consumed by any downstream model.
- **`recompute_temperature_fast.py`** — Input: existing weekly CSV + raw IMD data. Output: `imd_district_weekly_weather_2000_2024_FIXED.csv`. Status: working. Exists specifically to fix the coarse-temperature-grid zonal-stats bug without needing to redownload or recompute rainfall.
- **`compute_district_population.py`** — Input: boundaries + WorldPop rasters. Output: `district_population_2000_2020.csv`. Status: working. NOTE: originally run by Claude directly in a sandbox lacking `geopandas`/`rasterio`/`rasterstats`; uses `tifffile`+`PIL`+`matplotlib.path.Path` as a substitute — a DIFFERENT, less standard implementation than the hexagon-level version, worth knowing if debugging.
- **`disaggregation_prototype.py`** — Purely synthetic data, proves the method mechanically. NOT used for any real output. Status: complete, serves its proof-of-concept purpose, should not be modified/mistaken for the real model.
- **`wire_disaggregation_model.py`** — Input: H3 grid, hex population, dengue CSV, malaria CSV. Output: `dengue_hex_annual.csv`, `malaria_hex_annual.csv`, `disaggregation_fit_report.txt`. Status: COMPLETE, validated, this is the CURRENT real deliverable. Contains: state/district name normalization (with an explicit alias dictionary), Jammu & Kashmir/Ladakh pre-2019 split handling, the Poisson regression fitting logic, mass-preserving rescale, and extensive built-in validation/diagnostic printing. Known past bugs, all fixed (see PART 13): BOM/encoding KeyError, memory-heavy tuple-based grouping, a name-collision-induced mass-preservation failure.

**Important functions worth knowing:**
- `interpolated_population_column()` / `interpolate_population()` — vectorized (NOT row-wise `apply()`) linear interpolation of hexagon population between WorldPop anchor years.
- `normalize_state()` — case/whitespace normalization + explicit alias dictionary (`STATE_NAME_FIXES`) for known state-name spelling variants across data sources.
- `fit_and_disaggregate()` — the generic, reusable core function in `wire_disaggregation_model.py`; takes any case-count long-table + unit columns and returns hexagon-level mass-preserving predictions. Designed to be reusable for adding Tier 2 covariates later.

---

## PART 11 — IMPLEMENTATION STATUS TABLE

| Component | Status | Details | Next Action |
|---|---|---|---|
| Dengue data (state/annual) | DONE | OpenDengue, verified, 2010-2024 | None — ready for use |
| Malaria data (district/annual) | DONE | NCVBDC PDF, major bug found+fixed | None — ready for use |
| Rainfall/temperature (district/weekly) | DONE | IMD, extensively debugged, validated | Not yet consumed downstream |
| Population (district) | DONE | WorldPop, 5 snapshots | None |
| Population (hexagon) | DONE | WorldPop, 5 snapshots, fast method | None |
| Administrative boundaries | DONE | 724 districts cleaned | None |
| H3 grid (resolution 7) | DONE | 620,742 hexagons, validated | None |
| Spatial disaggregation model | DONE | Poisson regression, mass-preserving, population-only covariate | Could be extended with Tier 2 covariates |
| Tier 2 covariates (NDVI/land cover/water) | PLANNED | Sources verified accessible, nothing pulled | Pull if chosen as next step |
| Temporal disaggregation (annual->weekly) | PLANNED | Identified as necessary, not designed | Design + build |
| Feature engineering | PLANNED | Planned per original blueprint | Not started |
| Forecasting models (LightGBM/XGBoost) | PLANNED | Planned per original blueprint | Not started |
| Hotspot detection (Gi* on predicted risk) | PLANNED | Planned per original blueprint | Not started |
| Future hotspot validation | PLANNED | Planned per original blueprint | Not started |
| Dual-disease comparison | PLANNED | Planned per original blueprint | Not started |
| Explainability (SHAP) | PLANNED | Planned per original blueprint | Not started |
| Uncertainty/calibration | PLANNED | Marked "Partial" in original blueprint's own self-assessment | Not started |
| Early warning engine + dashboard | PLANNED | Planned per original blueprint | Not started |
| Research evaluation | PLANNED | Planned per original blueprint | Not started |
| Disaggregation validation against Bhopal ward data | PLANNED | Data collected, validation not run | Open task |
| GitHub repo structure | IN PROGRESS | Created, structured, Phase 4 committed; Phase 5 files handed to user for commit, commit not confirmed | Confirm commit |
| README / formal documentation | PLANNED | Not written | Not started |
| Research paper | UNKNOWN | UNKNOWN/NOT CONFIRMED whether one is required | Clarify with user/faculty expectations |

---

## PART 12 — CURRENT STATE (AT TIME OF THIS HANDOVER)

- **What is completely finished:** All Tier 1 mandatory datasets (6 datasets); the H3 spatial grid (620,742 hexagons); per-hexagon population; the real spatial disaggregation model for both diseases (population-only covariate, annual resolution).
- **What is currently being worked on:** Nothing actively mid-task — the disaggregation model work just concluded successfully. The user was in the process of committing Phase 5 (disaggregation) artifacts to GitHub and had just hit and fixed a local script bug (BOM/encoding `KeyError` in `wire_disaggregation_model.py`, now fixed).
- **What was the last successful action:** Claude tested the fixed (BOM-safe) version of `wire_disaggregation_model.py` end-to-end in its own sandbox against the real project files and confirmed it reproduces the same correct results (dengue: 7,796,424-row hex-year table, 451 unit-years, mass-preservation exact). The user was told to replace their local script and rerun.
- **What was the last failed step (now fixed):** A `KeyError: 'h3_index'` when merging the hexagon-population CSV, traced to a likely BOM character from a Windows CSV save — fixed by adding `encoding="utf-8-sig"` and column-name stripping to all `pd.read_csv()` calls in the script.
- **Files most recently changed:** `wire_disaggregation_model.py` (BOM fix, most recent); before that, `malaria_district_2000_2024.csv` (state-mislabeling fix — this OVERWROTE the previous version).
- **Current blocker:** NONE technical — the user needs to (1) rerun the BOM-fixed `wire_disaggregation_model.py` locally to confirm it completes on their machine, (2) commit Phase 5 artifacts to GitHub, then (3) decide the next phase direction (Tier 2 covariates vs. temporal disaggregation/feature engineering).
- **What should happen next (in order):**
  1. User confirms local run of `wire_disaggregation_model.py` completes successfully.
  2. Commit Phase 5 files to the GitHub repo: `wire_disaggregation_model.py` -> `src/disaggregation/`; corrected `malaria_district_2000_2024.csv` -> `data/processed/` (REPLACING the old buggy version); `disaggregation_fit_report.txt` -> `data/processed/`; `disaggregation_real_results_2024.png` -> `outputs/figures/` (new folder). Add `data/processed/*_hex_annual.csv` to `.gitignore`.
  3. Decide: Tier 2 covariates next, or move to temporal disaggregation/feature engineering. Claude's last recommendation leaned toward temporal/features first, but left the choice open.
- **What should NOT be done yet:** Do not start feature engineering or forecasting model work before either (a) Tier 2 covariates are pulled and incorporated, if that's the chosen path, or (b) temporal disaggregation is designed — the current hex-annual outputs are ANNUAL, not weekly.
- **Current phase:** Phase 5 (Spatial Disaggregation) — essentially complete, pending final commit confirmation.
- **Phase immediately after:** Phase 6 (Tier 2 covariates) OR Phase 7 (Temporal disaggregation) — order not yet decided, open decision.

---

## PART 13 — DEBUGGING HISTORY (DETAILED)

### 1. IMD weather download: connection timeouts
- **Problem:** `imdpune.gov.in` intermittently refuses connections (`ConnectTimeout`) under repeated requests.
- **Investigation:** Confirmed server-side flakiness, not a code bug.
- **Fix:** Rewrote the download loop to process ONE YEAR AT A TIME with up to 5 retries and a 45-second wait between attempts; added resume logic.
- **Status:** Resolved — all 25 years x 3 variables eventually downloaded successfully across multiple reruns.

### 2. IMD weather: resume logic pointed at the wrong folder (twice)
- **Problem:** After adding resume logic, reruns still redownloaded already-present years.
- **Cause (round 1):** Resume check looked in `imd_raw_data/{variable}/`, but the FIRST pre-fix run saved to `imdlib`'s default location, `./{variable}/`.
- **Fix (round 1):** Changed `DATA_DIR` to `"."` — this turned out WRONG (based on ambiguous user answer).
- **Fix (round 2, final):** Made the resume check search BOTH candidate locations, plus a consolidation step copying stray files into the canonical location.
- **Status:** Resolved.

### 3. IMD weather: leftover duplicate-download code
- **Problem:** After all years finished, the script redownloaded all 25 years of tmin again.
- **Cause:** A leftover code fragment from an earlier version of the script that survived an incomplete edit.
- **Fix:** Removed the leftover call.
- **Status:** Resolved.

### 4. Temperature data: 66-72% null (the major weather bug)
- **Problem:** 478 of 724 districts (later 72.3% of rows after an interim fix attempt) had 100% null Tmax/Tmin.
- **Cause (round 1):** `rasterstats.zonal_stats()`'s default only counts a pixel if its CENTER falls inside a polygon. Temperature's grid is 1x1 degree (~100km), far coarser than most districts, so most districts had ZERO qualifying pixels.
- **Attempted fix (round 1):** Added `all_touched=True` to the per-day call — extremely SLOW (8 hours) because it re-rasterizes from scratch every one of ~9,125 days.
- **Attempted fix (round 2, WRONG):** Single global `rasterize()` "zone raster" reused across days. Made it WORSE (72.3% null) because `rasterize()` assigns each pixel to exactly ONE winning district, and coarse pixels shared by many small districts only kept one.
- **Final correct fix (round 3):** Per-district pixel membership built INDEPENDENTLY (allowing a pixel to belong to multiple districts' membership lists), precomputed once, then `np.bincount` per day for fast aggregation.
- **Verification:** Final null rate 0.4%, explained edge cases only (Andaman/Nicobar islands, Anjaw border district). Zero Tmax<Tmin violations, zero duplicates, Delhi May 2022 heatwave spot-check (43.2C) matched real records.
- **Status:** Resolved, thoroughly validated.

### 5. Hexagon population: rasterstats too slow (same bug class as #4, caught proactively)
- **Problem:** Projected 50+ min/year x 5 years = 4+ hours (user reported 52 minutes elapsed on year 2000 alone).
- **Fix:** Same solution pattern as temperature's final fix — bulk `rasterize()` + `np.bincount`. LESS risky here because population's raster (1km) is FINER than hexagons (~2.3km), so "winner takes all" only misattributes thin boundary slivers, not whole hexagons.
- **Verification:** 0.15% national-level error, 75% of districts within 1.2%.
- **Status:** Resolved.

### 6. H3 grid: district-coverage validation bug
- **Problem:** Validation reported only 718/724 districts covered; one district ("Leh, Ladakh") showed an implausible 26,550 hexagons.
- **Cause:** Coverage/fallback logic used district NAME alone as the key, not (state, district). 5 district names repeat across different states nationally.
- **Fix:** Rekeyed all validation/fallback logic to (state, district) tuples.
- **Verification:** Final run: 724/724 pairs covered, 36/36 states, 0 duplicate hexagon IDs, visually confirmed correct India shape.
- **Status:** Resolved.

### 7. Malaria PDF parsing: Andaman & Nicobar page-break label loss (EARLY bug, first instance of this bug class)
- **Problem:** Initial parse produced only 24 distinct years (should be 25) with one year showing ~double the expected row count.
- **Cause:** On one page-break, the "Andaman And Nicobar Islands" state-label cell's text was lost due to a collision with a multi-line district name wrapping across the same rows, causing year-boundary detection to miss that transition.
- **Fix:** Added fallback detection using a known finite list of district names that ONLY belong to Andaman & Nicobar Islands.
- **Verification (at the time):** All 25 years present, smooth row-count growth, zero duplicates, spot-check matched raw PDF text, plausible national trend.
- **Status:** Resolved AT THE TIME — but see bug #8, which reveals this failure mode was much more widespread.

### 8. Malaria data: SYSTEMIC state-mislabeling (discovered later, during disaggregation work — the MOST SIGNIFICANT data-quality finding in the project)
- **Problem:** Cross-checking malaria rows against the trusted H3 grid revealed **52% of all 16,781 rows had a (state, district) pair that doesn't actually exist**.
- **Why earlier validation missed this:** National-level yearly totals remain correct regardless of which STATE a district's cases get attributed to.
- **Investigation:** Same root cause as bug #7, but SYSTEMATIC: small states (Sikkim: 4 districts, Lakshadweep: 1 district) are disproportionately affected because with so few rows, the state label's position lands very close to the boundary with the NEXT state's block. Recurred at multiple specific transitions in almost every year (e.g., "Tripura" bleeding forward to cover Uttar Pradesh's ~70+ districts in most years).
- **Fix:** Rebuilt STATE assignment using a canonical district->state lookup from the TRUSTED H3 grid, instead of trusting the fragile in-PDF label.
  - Unambiguous single-state district names: reassigned automatically (6,290 rows corrected).
  - 5 nationally-ambiguous district names (Aurangabad, Balrampur, Bilaspur, Hamirpur, Pratapgarh): kept original label if valid (103 rows); flagged `UNRESOLVED_ambiguous` if not (126 rows — documented residual).
  - District name spelling/transliteration alias table added (Kachchh->Kutch, Ahmadabad->Ahmedabad, etc.) — a SEPARATE issue from the mislabeling bug.
  - Rows whose district name doesn't exist anywhere in the grid (2,310 rows): left as-is, mostly explained by genuine historical district reorganizations (Sikkim's 2021 renaming, Andhra Pradesh's 2022 split) or special "#"-marked surveillance sub-units that aren't real districts.
- **Verification:** Invalid rate dropped from 52% to 14.2%, residual explained.
- **Status:** Substantially resolved; 14.2% residual documented, not actively chased further. **This correction OVERWROTE the previous `malaria_district_2000_2024.csv`** — any earlier copy elsewhere is STALE.

### 9. Disaggregation script: out-of-memory crash
- **Problem:** Malaria model-fitting step got silently killed (OOM) partway through testing.
- **Cause:** Grouping logic built a pandas column of Python TUPLE objects across millions of rows — heavy memory overhead, combined with not freeing memory between dengue and malaria fits.
- **Fix:** Replaced with `pandas.groupby(cols).ngroup()`, added explicit `del`/`gc.collect()` between fitting stages.
- **Status:** Resolved — full real-data run completed successfully, verified by Claude directly in its own sandbox (this script only needs pandas/numpy/scipy).

### 10. Disaggregation script: mass-preservation check failed (discrepancy of 2036)
- **Problem:** Built-in validation showed nonzero discrepancies between predicted and observed unit totals.
- **Cause:** State-name normalization mapped BOTH "Dadra and Nagar Haveli" and "Daman and Diu" to the same modern merged-UT name — correct post-2020-merger, but WRONG for earlier years where they were separate territories with separate case counts. Collapsing the names caused a collision that `.first()` silently resolved by discarding one value.
- **Fix:** Changed to explicitly SUM colliding case-count rows rather than silently keeping only the first.
- **Status:** Resolved — final result: exact mass-preservation (0.000000 max deviation) for both diseases.

### 11. Disaggregation script: `KeyError: 'state'` at save time
- **Problem:** After the grouping rewrite (#9), final CSV-save crashed because `state`/`district` display columns had been dropped earlier for memory efficiency.
- **Fix:** Explicitly carry `state`/`district` alongside their normalized versions through the pipeline.
- **Status:** Resolved.

### 12. Disaggregation script: BOM/encoding `KeyError: 'h3_index'` (MOST RECENT bug, just fixed)
- **Problem:** On the user's Windows machine, merging the hexagon-population CSV failed with `KeyError: 'h3_index'` despite the column visibly existing.
- **Cause (diagnosed, most likely explanation):** A UTF-8 byte-order-mark (BOM) character prepended to the first column name by a Windows tool, making the literal name `"\ufeffh3_index"` rather than `"h3_index"`.
- **Fix:** Added `encoding="utf-8-sig"` (strips a BOM if present, no-op if not) plus `.columns.str.strip()` to every `pd.read_csv()` call.
- **Verification:** Claude re-ran the fixed script end-to-end in its own sandbox against the real project files, confirmed identical correct results.
- **Status:** Fix shipped; user's own local confirmation NOT yet received as of this handover — this is the very next expected step.

### 13. Miscellaneous path-resolution issues (multiple scripts, multiple rounds)
- **Problem:** Several scripts initially assumed all files sit in the same flat folder, which broke once the project was organized into subfolders.
- **Fix:** All scripts rewritten to resolve paths via `Path(__file__).resolve().parent` climbing to a known project-root-relative structure.
- **Status:** Resolved, applied consistently.

### 14. Sandbox-specific issue (Claude's own environment, not the user's — informational only)
- Claude's own sandbox lacks `geopandas`/`rasterio`/`rasterstats`/`GDAL`/`h3` and has no network access to install them. For district-level population work, Claude built a pure-Python substitute: `tifffile` (GeoTIFF tags) + `PIL`/`Pillow` (pixel decoding — `tifffile`'s own decoder failed on this data's LZW compression) + `matplotlib.path.Path` (vectorized point-in-polygon zonal statistics). Documented as reusable for any future raster-covariate work Claude does directly in its own sandbox.

---

## PART 14 — DECISION LOG

**DECISION: Pivot from "H3 as native prediction resolution" to "statistical disaggregation"**
STAGE: Early, after real-data resolution investigation
REASON: Real Indian case data is only available at state/district resolution
ALTERNATIVES: (a) abandon H3, stay district-level; (b) naive even-split
WHY REJECTED: (a) eliminates fine-resolution ambition; (b) statistically indefensible, proven inferior in synthetic test
STATUS: Finalized, implemented, validated

**DECISION: H3 resolution 7 (not 6)**
STAGE: After initial H3 grid planning
REASON: User explicitly rejected "district-wise dressed up as hexagons"; resolution 7 close to the ceiling of what real covariates support
ALTERNATIVES: Resolution 6 (initially proposed), resolution 8 (considered, rejected)
STATUS: Finalized, implemented (620,742 actual hexagons)

**DECISION: Population as sole current disaggregation covariate; weather excluded from the SPATIAL disaggregation step**
STAGE: During real disaggregation model design
REASON: Spatial disaggregation explains WITHIN-district variation; weather varies too little within most districts; population is the standard covariate in comparable real-world work
ALTERNATIVES: Include weather in the spatial model too
WHY REJECTED (for now): Would conflate spatial and temporal/seasonal questions; weather earmarked for the not-yet-built temporal stages
STATUS: Finalized for the current model; not a permanent exclusion

**DECISION: Always join on (state name, district name), never `dt_code` alone**
STAGE: During boundary-file cleaning
REASON: `dt_code` found non-unique nationally, other quality issues
STATUS: Finalized, applied consistently

**DECISION: Fix malaria state-mislabeling via canonical lookup, not PDF re-parsing**
STAGE: During disaggregation work, upon discovering the 52% bug
REASON: District names and case/death NUMBERS were still correct; lookup-based correction more tractable than fixing fragile PDF state-tracking
STATUS: Finalized, 14.2% residual accepted as documented limitation

**DECISION: Treat hex-annual CSVs as local/regenerable, not repo/chat-transferable**
STAGE: After the real run produced 409MB/748MB outputs
REASON: Too large for GitHub or chat
STATUS: Finalized

**OPEN DECISION: Tier 2 covariates next, vs. temporal disaggregation/feature engineering next**
STATUS: NOT YET DECIDED — user was asked this exact question at the end of the last session; no answer captured yet.

---

## PART 15 — FAILED OR REJECTED APPROACHES

**APPROACH: Naive even-split disaggregation**
WHY CONSIDERED: Simplest possible baseline
WHAT WAS TRIED: Comparison baseline in the synthetic prototype test
WHAT HAPPENED: 74.6% hotspot overlap vs. 86.7% for the real method
WHY REJECTED: Demonstrably inferior, conceptually indefensible
TRY AGAIN?: No — serves its purpose as a baseline reference.

**APPROACH: Single global `rasterize()` "zone raster" at COARSE resolution**
WHY CONSIDERED: Avoid slowness of per-feature `zonal_stats()`
WHAT WAS TRIED: One bulk rasterize call, one winning district per pixel
WHAT HAPPENED: Made the temperature null-data problem WORSE (72.3% vs 66%)
WHY REJECTED: Structurally wrong when the raster is COARSER than the polygons
TRY AGAIN?: No for coarse cases, but the same TECHNIQUE works fine at FINE resolution (hexagon population) — resolution-dependent lesson, not a blanket rejection.

**APPROACH: Python-tuple-based pandas grouping at multi-million-row scale**
WHY CONSIDERED: Straightforward composite group key
WHAT WAS TRIED: `list(zip(...))` as a DataFrame column + dict mapping
WHAT HAPPENED: Out-of-memory crash on the malaria table
WHY REJECTED: Heavy per-row memory overhead at scale
TRY AGAIN?: No — `groupby().ngroup()` is strictly better, now used consistently.

**APPROACH: Row-wise `DataFrame.apply()` for population interpolation**
WHY CONSIDERED: Simple to write
WHAT WAS TRIED: Per-row Python interpolation across 620k+ hexagons, repeated per year
WHAT HAPPENED: Likely major contributor to slowness/OOM issues
WHY REJECTED: Vectorized numpy is orders of magnitude faster
TRY AGAIN?: No — already replaced.

**APPROACH: Re-parsing the malaria PDF to fix state-mislabeling at the source**
WHY CONSIDERED: Would be the "purest" fix
WHAT WAS TRIED: Not attempted — considered and passed over
WHY REJECTED: Canonical-lookup post-hoc fix judged more tractable and sufficient
TRY AGAIN?: Possibly if the 14.2% residual becomes a real problem later — not currently planned.

---

## PART 16 — EXPERIMENTS

### Experiment 1: Synthetic disaggregation prototype, v1 (weak result)
- **Dataset:** Synthetic — 25 districts, correlated covariates (population/NDVI-like deliberately correlated)
- **Configuration:** Poisson regression, `Nelder-Mead`
- **ACTUAL MEASURED RESULT:** Spearman correlation = 0.622 (naive baseline 0.582); top-10% hotspot overlap = 55.6% (naive 51.1%)
- **Interpretation:** Weak improvement, traced to identifiability problems
- **Decision:** Redesign with more districts, decorrelated covariates

### Experiment 2: Synthetic disaggregation prototype, v2 (strong result)
- **Dataset:** Synthetic — 225 districts, decorrelated covariates, standardized
- **Configuration:** Poisson regression, `L-BFGS-B`
- **ACTUAL MEASURED RESULT:** Fitted coefficients closely recovered true generating parameters (e.g. true intercept -11.0 vs fitted -10.96; true population effect 0.00035 vs fitted 0.00035). Spearman correlation = 0.919 (naive 0.898); top-10% hotspot overlap = 86.7% (naive 74.6%)
- **Decision:** Proceed to real data with confidence in the method

### Experiment 3: Hexagon population cross-validation
- **Dataset:** Real — 620,742 hexagons x 5 years, aggregated to district level, compared against `district_population_2000_2020.csv`
- **ACTUAL MEASURED RESULT:** National total (2020): 1,404,944,653 (hex-summed) vs 1,402,908,487 (known) — 0.15% difference. 75% of districts within 1.2% error.
- **Decision:** Method accepted as accurate enough for use as the disaggregation covariate

### Experiment 4: Real dengue disaggregation model fit
- **Dataset:** Real — OpenDengue state-annual (2010-2024, 451 state-years after J&K/Ladakh duplication and Dadra/Daman collision-summing) x 620,742 hexagons' interpolated population
- **ACTUAL MEASURED RESULT (from the confirmed BOM-fixed rerun):** Intercept = -10.5241, log-population effect = +0.8652 (POSITIVE). Mass-preservation: max deviation = 0.000000 across all 451 unit-years.
- **Interpretation:** Consistent with dengue being urban/*Aedes*-associated
- **Note:** Treat `disaggregation_fit_report.txt` as the authoritative source for exact figures, not this summary.

### Experiment 5: Real malaria disaggregation model fit
- **Dataset:** Real — NCVBDC district-annual (2000-2024, 14,402 district-years) x hexagon population, AFTER the state-mislabeling fix
- **ACTUAL MEASURED RESULT:** Population coefficient NEGATIVE (exact value in `disaggregation_fit_report.txt` — only qualitatively confirmed as negative in the captured conversation, not precisely re-quoted here to avoid transcription error). Mass-preservation: max deviation = 0.000000 across all 14,402 unit-years.
- **Interpretation:** Consistent with malaria's real-world rural/forest-fringe concentration
- **Visual validation:** 2024 map showed dengue concentrated in Delhi-NCR/Mumbai/Chennai/Kolkata/Kerala coast; malaria concentrated in Odisha-Chhattisgarh-Jharkhand belt + Northeast. Total predicted 2024 national cases (ACTUAL MEASURED): dengue ~233,519; malaria ~34,538.

**No other experiments (feature engineering, forecasting, hotspot detection, ablations, baselines) have been run — those stages haven't been built yet.**

---

## PART 17 — MODELING

- **Models actually implemented:** ONE custom type — Poisson regression with an aggregation (mass-preservation) constraint, `scipy.optimize.minimize` (`L-BFGS-B`). This is the SPATIAL DISAGGREGATION model, not a forecasting model. Two independently-fit instances: dengue (state-level), malaria (district-level).
- **LightGBM, XGBoost, Random Forest, LSTM, ensembles, statistical baselines:** ALL still PLANNED, NONE implemented/tested/scaffolded. No hyperparameters chosen.
- **Current role of the implemented model:** Produces the hexagon-level ANNUAL case surface intended to feed the not-yet-built forecasting stage.
- **Feature set used:** Exactly one real covariate — hexagon population (standardized `log1p`), used both as regression covariate and multiplicative exposure offset.
- **Validation strategy:** Mass-preservation (exact, numerically confirmed) + population covariate cross-validation + qualitative/visual epidemiological sanity-check. NOT a held-out predictive validation in the usual ML sense (no hexagon-level ground truth exists to hold out against — inherent to the problem, not a gap).

---

## PART 18 — EVALUATION AND VALIDATION

**What HAS been validated (all ACTUAL):**
- Mass-preservation: exact (0.000000 deviation), both diseases, all unit-years.
- Population covariate accuracy: 0.15% national error vs independently-computed district population.
- Weather data physical plausibility: zero Tmax<Tmin violations, zero duplicates, realistic ranges, Delhi May 2022 heatwave spot-check matched.
- Malaria/dengue name-matching validity: 52%->14.2% invalid rate improvement, systematically checked.
- Qualitative/visual epidemiological plausibility: dengue urban / malaria rural-forest patterns match known geography.

**What has NOT been validated / not yet implemented:**
- Disaggregation validated against actual finer-than-district ground truth (Bhopal ward data collected, not yet used).
- Any forecasting model's predictive accuracy (none exists yet).
- Any hotspot detection accuracy (Gi* not implemented).
- The original blueprint's full Research Evaluation layer: baselines, ablations (including E3, H3-vs-admin-adjacency), spatial/temporal hold-out validation, hotspot method comparison — NONE implemented for the real pipeline.
- Leakage prevention — no feature engineering exists yet, so no leakage risk yet either, but documented as a future requirement.
- Calibration — not started.

---

## PART 19 — EXPLAINABILITY

- **SHAP:** PLANNED per original blueprint. NOT implemented — no trained forecasting model exists yet.
- **Feature importance:** Not applicable yet (only 2 coefficients in the current disaggregation model, already directly interpreted).
- **How it fits the research contribution:** Per the original blueprint, intended for per-cell and per-hotspot interpretability once the forecasting model exists.

---

## PART 20 — UNCERTAINTY

- **Prediction uncertainty/confidence intervals:** DISCUSSED ONLY (self-assessed "Partial" even in the original blueprint's own planning), not implemented.
- **Model/data/spatial uncertainty:** Not formally addressed; known limitations are documented qualitatively (see PART 29) but not statistically quantified.
- **Calibration:** Not implemented, not designed.
- **Status: PLANNED or DISCUSSED ONLY for all — none IMPLEMENTED, none formally REJECTED.**

---

## PART 21 — VISUALIZATIONS AND OUTPUTS

| Output | Purpose | Status | Intended use |
|---|---|---|---|
| `disaggregation_validation.png` | 3-panel comparison: hidden truth vs model vs naive (SYNTHETIC) | Generated | Proof-of-concept evidence |
| H3 grid visual check (India map by state) | Sanity-check real grid | Generated | Verification / methods figure |
| `disaggregation_real_results_2024.png` | Real 2024 dengue vs malaria risk maps | Generated | Strong candidate results figure |
| Risk maps (general) | PLANNED | Not built | Dashboard/paper |
| Hotspot maps | PLANNED | Not built | Core results |
| Forecast graphs | PLANNED | Not built | Results |
| SHAP plots | PLANNED | Not built | Explainability |
| Evaluation charts (P/R/F1/IoU/lead-time) | PLANNED | Not built | Likely strongest evidentiary figures |
| Dashboard | PLANNED | Not built | Final deliverable |
| `docs/brain.md` | Running plain-text log | Exists, kept updated | Historical record, now largely superseded by this handover |

---

## PART 22 — ARTIFACT INVENTORY

| Artifact | Purpose | Location/Status | Notes |
|---|---|---|---|
| `Dengue_Malaria_Research_Blueprint.docx` | Original vision document (user-uploaded) | Uploaded at project start | Source of original pipeline diagram and novelty framing |
| `dengue_state_2010_2024.csv` | Cleaned dengue data | `data/processed/` | Also referred to as `filtered_data_SEARO_....csv` (original upload filename) — same data |
| `malaria_district_2000_2024.csv` | Cleaned, CORRECTED malaria data | `data/processed/` | **OVERWRITTEN during the bug fix — any older copy is STALE** |
| `imd_district_weekly_weather_2000_2024_FIXED.csv` | Validated weekly weather | `data/processed/` | "_FIXED" distinguishes from an earlier buggy version |
| `district_population_2000_2020.csv` | District-level population | `data/processed/` | |
| `hex_population_2000_2020.csv` | Hexagon-level population | `data/processed/` | THE key disaggregation covariate |
| `india_districts_clean.geojson` | Cleaned 724-district boundaries | `data/boundaries/` | |
| `india_h3_grid_res7.csv` | 620,742-hexagon grid | `data/processed/` | |
| `dengue_hex_annual.csv` | Disaggregated dengue predictions | Local only, gitignored, 409MB | Regenerable |
| `malaria_hex_annual.csv` | Disaggregated malaria predictions | Local only, gitignored, 748MB | Regenerable |
| `disaggregation_fit_report.txt` | Fit log (coefficients, validation) | `data/processed/` | Authoritative source for exact coefficients |
| `bhopal_wards.geojson` | 86 Bhopal wards, Tier 3 validation asset | Exact current repo location not confirmed | NOT YET USED |
| `disaggregation_prototype.py` | Synthetic proof of concept | `src/disaggregation/` | Not for production use |
| `wire_disaggregation_model.py` | Real disaggregation model | `src/disaggregation/` | Current core deliverable |
| Various `src/data_prep/*.py` | Data collection/cleaning pipeline | `src/data_prep/` | See PART 10 |
| `docs/brain.md` | Running project log | `docs/` | Predates/overlaps this handover |

---

## PART 23 — RESEARCH PAPER PLAN

**UNKNOWN/NOT CONFIRMED in its entirety.** No formal research paper plan, title, abstract, or section outline has been discussed or drafted. The original "Research Blueprint" document implies academic framing, but nothing concrete beyond that exists.

**Reasonable future paper components (NOT a confirmed plan, just a convenience reconstruction):**
- Figures: the three visualizations already generated are strong candidates; future hotspot-validation results would likely be central evidence.
- Claims with current evidence: disaggregation method beats naive baseline (synthetic); real disaggregation produces sensible, mass-preserving results.
- Claims NOT yet supportable: anything about forecasting accuracy, hotspot detection performance, or dual-disease comparative findings.

---

## PART 24 — PROJECT PRESENTATION / DEMONSTRATION

UNKNOWN/NOT CONFIRMED — no discussion of a presentation, demo, or workflow diagram (beyond the original blueprint's own pipeline diagram) has occurred.

---

## PART 25 — UPCOMING ROADMAP

```
IMMEDIATE NEXT TASKS
  1. User confirms wire_disaggregation_model.py (BOM-fixed) runs
     successfully locally
  2. Commit Phase 5 artifacts to GitHub
        |
OPEN DECISION POINT (not yet resolved)
  Tier 2 covariates (NDVI/land cover/water) vs temporal disaggregation +
  feature engineering. Claude's last lean: temporal/features first, but
  explicitly left to the user.
        |
NEXT PHASE (whichever is chosen)
        |
FOLLOWING PHASE
  Feature engineering: lags, rolling stats, growth rates, H3 neighbor
  features (leakage-safe), seasonality encoding
        |
FINAL IMPLEMENTATION PHASE
  Forecasting models -> Hotspot detection (Gi* on predicted risk) ->
  Dual-disease comparison -> Explainability (SHAP) ->
  Uncertainty/calibration -> Early warning engine -> Dashboard
        |
VALIDATION PHASE
  Future-hotspot validation -> Research evaluation layer (baselines,
  ablations, spatial/temporal holdout, hotspot method comparison)
        |
RESEARCH PAPER / FINAL PRESENTATION
  UNKNOWN/NOT CONFIRMED whether formally required
```

---

## PART 26 — CURRENT PHASE DEFINITION

- **CURRENT PHASE:** Phase 5 — Spatial Disaggregation (real data)
- **PHASE OBJECTIVE:** Disaggregate real state/district-level case totals to H3-hexagon resolution using real covariates, exactly mass-preserving, validated.
- **COMPLETED WITHIN PHASE:** Model design/implementation; major data-quality bug found+fixed; full real-data run completed and validated; a subsequent local-environment bug found+fixed.
- **CURRENT TASK:** Confirming the BOM-fixed script runs on the user's machine, then committing to GitHub.
- **CURRENT BLOCKER:** None technical — awaiting user confirmation/commit and Phase 6 decision.
- **PHASE EXIT CRITERIA:** Phase 5 artifacts committed; Phase 6 direction chosen.
- **NEXT PHASE:** Phase 6 — Tier 2 Covariates OR Temporal Disaggregation (not yet decided).

---

## PART 27 — "DO NOT BREAK THIS" SECTION

1. **H3 resolution = 7.** Deliberate, user-driven, tied to the core novelty claim. Do not revert to coarser resolution without explicit user request.
2. **Always join geographic data on (state name, district name), never `dt_code` alone.**
3. **`malaria_district_2000_2024.csv` is the STATE-MISLABELING-CORRECTED version.** Never reintroduce an older uncorrected copy.
4. **The mass-preserving rescale is exact by construction, numerically verified.** Do not simplify this away.
5. **`dengue_hex_annual.csv`/`malaria_hex_annual.csv` must NOT be committed to GitHub or routinely passed through chat.**
6. **Population is used BOTH as offset AND covariate** — deliberate, not an oversight.
7. **Weather is currently NOT part of the spatial disaggregation model** — deliberate, not a gap to hastily fix.
8. **All scripts use `Path(__file__).resolve().parent`-relative path resolution.** Preserve this pattern.
9. **No synthetic data in the final/real pipeline.** `disaggregation_prototype.py` is proof-of-concept only.
10. **No individual-technique novelty claims** (LightGBM/XGBoost/SHAP/Gi*/climate features) — novelty is the combination + disaggregation-validation methodology.

**OPEN DECISIONS (safe to discuss/change):**
- Tier 2 covariates vs temporal disaggregation as next phase
- Exact temporal disaggregation method/design (not designed at all yet)
- Whether a formal research paper is required
- Exact forecast holdout scheme

---

## PART 28 — OPEN QUESTIONS

1. **Tier 2 covariates next, or temporal disaggregation/feature engineering next?** Determines immediate next work. No answer captured yet.
2. **Should Bhopal ward data be used to validate the disaggregation model now, or later?** The one piece of real finer-than-district ground truth available; would substantially strengthen the rigor claim. No preference stated.
3. **Is a formal written research paper required?** Unknown — never discussed.
4. **How should the residual 14.2% invalid malaria rows be handled going forward?** Implicitly "accepted as documented limitation," not explicitly finalized as policy.
5. **Should the population-only disaggregation model be preserved as a baseline for future ablations, or just replaced when Tier 2 covariates are added?** Not discussed, but the original blueprint's own Research Evaluation layer implies ablations (with/without covariates) are part of the intended methodology — suggesting preservation as a baseline is likely appropriate.

---

## PART 29 — RISKS AND LIMITATIONS

- **Data limitations:** Case data is annual-only (hard ceiling without temporal disaggregation); malaria has a 14.2% residual mislabeling rate; population is currently the ONLY real covariate; boundary data reflects claimed, not always actual-controlled, territory; no genuine hexagon-level case ground truth exists anywhere (inherent to the problem).
- **Technical limitations:** Hex-annual CSVs are very large, need efficient (Parquet/chunked) handling downstream; Claude's own sandbox lacks key geospatial libraries.
- **Research limitations:** No formal literature review conducted; no held-out predictive validation of the disaggregation model yet.
- **Computational limitations:** Hex-year tables for model fitting are large (millions to tens of millions of rows); vectorization/memory-efficiency now established as necessary.
- **Time constraints:** UNKNOWN/NOT CONFIRMED — no explicit deadline stated, though faculty review is the implied endpoint.
- **Reproducibility concerns:** Large raw files (IMD grids, WorldPop rasters) are gitignored and must be manually re-acquired for a full from-scratch reproduction — README should document this clearly when written.
- **Potential sources of bias:** "Winner takes all" pixel-boundary effect in fast rasterize-based zonal statistics (small, documented, bounded).
- **Potential leakage:** Not yet a concern (no temporal features exist yet) but will be a real risk once lag/rolling/neighbor features are built.
- **Spatial limitations:** Hexagon resolution capped by real covariate resolution (1km population).
- **Temporal limitations:** Annual-only case data, as above.
- **Generalization limitations:** UNKNOWN/NOT CONFIRMED — no discussion of generalizing beyond India or beyond dengue/malaria.

---

## PART 30 — EXACT HANDOVER STATE

```
==================================================
PROJECT HANDOVER SNAPSHOT
==================================================

PROJECT:
VectorHotspot -- Dengue-Malaria Dual-Disease Spatiotemporal Early Warning
System for India (GitHub: ThaufeeqAhamed/VectorHotspot, private)

RESEARCH OBJECTIVE:
Disaggregate coarse (state/district) Indian dengue and malaria surveillance
data to fine (H3 hexagon, resolution 7, ~620,742 cells) spatial resolution
using real covariates, in a mass-preserving/validated way, as the
foundation for (planned) weekly forecasting, hotspot detection, and
rigorous future-hotspot validation.

CURRENT METHODOLOGY:
Global Poisson regression per disease (population as covariate + exposure
offset), fit via aggregated log-likelihood against real state/district-
annual totals, hexagon predictions rescaled to exactly preserve known
totals.

COMPLETED:
1. All 6 Tier 1 mandatory datasets -- collected, cleaned, validated
2. H3 resolution-7 grid -- 620,742 hexagons, all districts/states covered
3. Per-hexagon population -- the key disaggregation covariate, validated
4. Real spatial disaggregation model -- both diseases, exact mass-
   preservation confirmed, epidemiologically sensible results confirmed

CURRENT PHASE:
Phase 5 (Spatial Disaggregation) -- functionally complete, pending final
GitHub commit confirmation

CURRENT TASK:
User confirming a just-fixed BOM/encoding bug in wire_disaggregation_model.py
runs correctly locally, then committing Phase 5 files to GitHub

LAST SUCCESSFUL ACTION:
Claude validated the BOM-fixed script end-to-end in its own sandbox against
the real project data, confirming correct results

CURRENT BLOCKER:
None technical. Awaiting (a) user's local confirmation, (b) user's decision
on Phase 6 direction

NEXT ACTION:
User reruns the fixed script, commits Phase 5 artifacts to GitHub, then
states which Phase 6 direction to pursue

NEXT 5 TASKS:
1. Confirm local run of wire_disaggregation_model.py succeeds
2. Commit Phase 5 files to GitHub (corrected malaria CSV REPLACING the
   old one; .gitignore updated for large hex_annual files)
3. Decide: Tier 2 covariates vs temporal disaggregation
4. Execute whichever is chosen
5. Use Bhopal ward data to validate the disaggregation model's
   within-district spatial pattern (currently unaddressed open task)

UPCOMING PHASES:
1. Tier 2 covariates OR Temporal disaggregation (order TBD)
2. Feature engineering
3. Forecasting models -> Hotspot detection -> Future-hotspot validation
   -> Dual-disease comparison -> Explainability -> Uncertainty/calibration
   -> Early warning engine -> Dashboard -> Research evaluation

IMPORTANT DECISIONS:
1. H3 resolution 7 (not 6) -- user-driven, tied to core novelty claim
2. Statistical disaggregation as the core spatial methodology
3. Population as sole current covariate; weather reserved for temporal
   stage

KNOWN FAILED APPROACHES:
1. Naive even-split disaggregation -- proven inferior
2. Single global rasterize() zone raster at COARSE resolution -- made
   temperature bug worse (fine at FINE resolution though)
3. Python-tuple-based pandas grouping at scale -- OOM

IMPORTANT FILES:
1. wire_disaggregation_model.py (src/disaggregation/) -- the real model
2. malaria_district_2000_2024.csv (data/processed/) -- CORRECTED version
3. india_h3_grid_res7.csv, hex_population_2000_2020.csv -- spatial
   foundation

IMPORTANT ARTIFACTS:
1. disaggregation_real_results_2024.png -- visual proof of real results
2. disaggregation_fit_report.txt -- authoritative coefficient values
3. Dengue_Malaria_Research_Blueprint.docx -- original vision document

OPEN QUESTIONS:
1. Tier 2 covariates next, or temporal disaggregation next?
2. Should Bhopal ward validation happen now or later?
3. Is a formal research paper required?

DO NOT CHANGE:
1. H3 resolution 7
2. The (state,district)-tuple join-key rule
3. The corrected malaria_district_2000_2024.csv
4. The exact mass-preservation guarantee
5. Path(__file__)-relative path resolution pattern

==================================================
END OF HANDOVER SNAPSHOT
==================================================
```

---

## PART 31 — INSTRUCTIONS FOR THE NEXT CLAUDE

- Treat this as an EXISTING, substantially-progressed project, not a new one. Do not propose starting over or re-litigating already-finalized decisions (PART 27) without the user explicitly asking.
- Read this entire handover document before proposing major changes, especially before touching H3 resolution, the join-key convention, the malaria data file, or the disaggregation model's core design.
- Do NOT restart completed work (Phases 1-5 are done and validated).
- Do NOT repeat failed approaches (PART 15) without a clearly new reason.
- Clearly distinguish completed / in-progress / planned work in every response.
- Verify actual repository/file state before assuming implementation status — do not assume something works just because this document says it's complete, without confirming the user's environment agrees where relevant.
- NEVER invent experiment results. Defer to `disaggregation_fit_report.txt` for exact figures rather than guessing.
- NEVER claim something is implemented unless you have direct evidence.
- Ask before major architectural changes (H3 resolution, covariate structure, repo restructuring).
- Prefer incremental changes, consistent with this project's established pattern of validating one phase at a time.
- Maintain reproducibility: new scripts follow the `Path(__file__)`-relative pattern; large outputs stay local/gitignored.
- Maintain consistency between code, methodology, and any future paper.
- Keep a running project state; update it after major tasks (this project used both a persistent memory file and `docs/brain.md` — continue this practice or ask the user how they want state tracked in the new account).
- Clearly state what changed and what remains after every substantial piece of work.
- Clearly identify blockers immediately, and be honest and specific about bugs — this project's history shows a strong pattern of thorough, honest bug diagnosis; continue that standard.

---

## PART 32 — SELF-AUDIT

1. Original idea captured? Yes — PART 1, 2.
2. Evolution captured? Yes — PART 2.
3. Current methodology captured? Yes — PART 6, 7, 8.
4. Technical decisions captured? Yes — PART 14.
5. Failed approaches captured? Yes — PART 15.
6. Completed work captured? Yes — PART 11, 12.
7. Current work captured? Yes — PART 12, 26.
8. Upcoming work captured? Yes — PART 25.
9. Research reasoning captured? Yes, with caveats where informal — PART 3, 4.
10. Codebase structure captured? Yes — PART 10.
11. Datasets captured? Yes — PART 9.
12. Experiments captured, actual vs interpretation distinguished? Yes — PART 16.
13. Results captured? Yes — PART 16, 21.
14. Artifacts captured? Yes — PART 22.
15. Open questions captured? Yes — PART 28.
16. Blockers captured? Yes — PART 12, 26 (none technical; pending user action/decision).
17. Important assumptions captured? Yes — e.g. population interpolation assumptions (PART 7).
18. Uncertain items identified? Yes, extensively marked UNKNOWN/NOT CONFIRMED throughout.
19. Could a new Claude continue using this? Yes — full data lineage, exact file locations/purposes, precise pipeline state, decision reasoning, and ordered next actions are all provided.
20. What's still missing (from the source conversation, not from this document's effort)? Formal research question wording; whether a written paper is required; presentation/demo plans; a literature review; exact deadline; Bhopal ward file's precise current repo location; precise malaria model coefficient values (qualitatively confirmed negative only — see `disaggregation_fit_report.txt` for exact figures).

---

*End of handover document. This file should be uploaded or pasted in full to the new Claude account/conversation to restore full project context.*
