#!/usr/bin/env python3
"""
Real-data spatial disaggregation model for VectorHotspot project.
Disaggregates state-level dengue and district-level malaria annual case counts
to H3 hexagon resolution using population as covariate, mass-preserving.

Based on the specifications in PROJECT_HANDOVER.md Phase 5.
"""

import pandas as pd
import numpy as np
from scipy import optimize
from pathlib import Path
import gc
import warnings

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# State name normalization — maps all known raw variants to the canonical
# name used in the H3 grid (india_districts_clean.geojson).
STATE_NAME_FIXES = {
    'A & N Island':                            'Andaman and Nicobar Islands',
    'A&N Island':                              'Andaman and Nicobar Islands',
    'Andaman And Nicobar':                     'Andaman and Nicobar Islands',
    'Andaman And Nicobar Islands':             'Andaman and Nicobar Islands',

    'D&N Haveli':                              'Dadra and Nagar Haveli and Daman and Diu',
    'D & N Haveli':                            'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra And Nagar Haveli':                  'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & Diu':                             'Dadra and Nagar Haveli and Daman and Diu',
    'Daman And Diu':                           'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra & Nagar Haveli':                    'Dadra and Nagar Haveli and Daman and Diu',
    'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',

    'J & K':                                   'Jammu and Kashmir',
    'Jammu & Kashmir':                         'Jammu and Kashmir',
    'Jammu And Kashmir':                       'Jammu and Kashmir',

    'Orissa':                                  'Odisha',
    'Pondicherry':                             'Puducherry',
    'Uttrakhand':                              'Uttarakhand',
}


def normalize_state(state_name):
    """
    Normalize a state name to the canonical form used in the H3 grid.
    Handles all-caps input, embedded whitespace/newlines, and known aliases.
    """
    if pd.isna(state_name):
        return state_name
    normalized = ' '.join(str(state_name).split()).title()
    return STATE_NAME_FIXES.get(normalized, normalized)


def interpolate_population(hex_pop_wide, target_years):
    """
    Vectorized population interpolation between WorldPop anchor years.

    Args:
        hex_pop_wide: WIDE-format DataFrame with h3_index + population_YYYY columns
        target_years: List of years to interpolate to

    Returns:
        LONG-format DataFrame with columns: h3_index, year, population
    """
    anchor_years = [2000, 2005, 2010, 2015, 2020]
    result_rows = []

    for target_year in target_years:
        if target_year in anchor_years:
            pop_col = f'population_{target_year}'
            year_data = hex_pop_wide[['h3_index', pop_col]].copy()
            year_data['year'] = target_year
            year_data['population'] = year_data[pop_col]
            result_rows.append(year_data[['h3_index', 'year', 'population']])
        elif target_year < 2000:
            year_data = hex_pop_wide[['h3_index', 'population_2000']].copy()
            year_data['year'] = target_year
            year_data['population'] = year_data['population_2000']
            result_rows.append(year_data[['h3_index', 'year', 'population']])
        elif target_year > 2020:
            year_data = hex_pop_wide[['h3_index', 'population_2020']].copy()
            year_data['year'] = target_year
            year_data['population'] = year_data['population_2020']
            result_rows.append(year_data[['h3_index', 'year', 'population']])
        else:
            lower_year = max([y for y in anchor_years if y <= target_year])
            upper_year = min([y for y in anchor_years if y >= target_year])
            alpha = (target_year - lower_year) / (upper_year - lower_year)
            lower_col = f'population_{lower_year}'
            upper_col = f'population_{upper_year}'

            year_data = hex_pop_wide[['h3_index', lower_col, upper_col]].copy()
            year_data['year'] = target_year
            year_data['population'] = (
                (1 - alpha) * year_data[lower_col] + alpha * year_data[upper_col]
            )
            result_rows.append(year_data[['h3_index', 'year', 'population']])

    return pd.concat(result_rows, ignore_index=True)


def fit_and_disaggregate(case_df, unit_cols, case_col, hex_pop_wide, hex_grid,
                         disease_name, report_file=None):
    """
    Generic disaggregation via Poisson regression with an aggregation constraint.

    Args:
        case_df: coarse (state/district) case counts
        unit_cols: e.g. ['state'] or ['state', 'district']
        case_col: column with case counts
        hex_pop_wide: WIDE-format hex population (h3_index + population_YYYY)
        hex_grid: h3_index -> district, state mapping
        disease_name: for logging/output naming
        report_file: optional file handle
    """
    print(f"\n=== {disease_name.upper()} DISAGGREGATION ===")
    if report_file:
        report_file.write(f"\n=== {disease_name.upper()} DISAGGREGATION ===\n")

    # Normalize state names
    if 'state' in case_df.columns:
        case_df = case_df.copy()
        case_df['state'] = case_df['state'].apply(normalize_state)

    hex_grid = hex_grid.copy()
    if 'state' in hex_grid.columns:
        hex_grid['state'] = hex_grid['state'].apply(normalize_state)

    # Combine pre-2019 J&K / Ladakh counts (one state in data before 2019)
    if 'state' in case_df.columns and 'year' in case_df.columns:
        jk_ladakh_mask = (
            (case_df['state'].isin(['Jammu and Kashmir', 'Ladakh'])) &
            (case_df['year'] < 2019)
        )
        if jk_ladakh_mask.any():
            print(f"Combining pre-2019 J&K/Ladakh counts for {jk_ladakh_mask.sum()} rows")
            case_df.loc[jk_ladakh_mask, 'state'] = 'Jammu and Kashmir'

    # Deduplicate / sum any colliding rows within the same unit-year
    # (e.g. Dadra & Nagar Haveli + Daman & Diu merged post-normalization,
    # or pre-2019 J&K + Ladakh re-labeled above)
    initial_rows = len(case_df)
    case_df = (case_df
               .groupby(unit_cols + ['year'], as_index=False)
               [case_col]
               .sum())
    collapsed = initial_rows - len(case_df)
    if collapsed > 0:
        print(f"Collapsed {collapsed} duplicate/colliding unit-year rows by summing")

    print("Building modeling dataset...")
    case_years = sorted(case_df['year'].unique())
    print(f"Case data years: {min(case_years)}-{max(case_years)} ({len(case_years)} years)")

    # Interpolate population (wide -> long) for the case years
    hex_pop_interp = interpolate_population(hex_pop_wide, case_years)

    # Join hex grid + interpolated population, then case data
    hex_data = hex_grid.merge(hex_pop_interp, on='h3_index', how='inner')
    modeling_data = hex_data.merge(case_df, on=unit_cols + ['year'], how='inner')

    print(f"Modeling dataset: {len(modeling_data):,} hexagon-year rows")
    print(f"Covers {modeling_data[unit_cols + ['year']].drop_duplicates().shape[0]:,} unit-years")
    if modeling_data.empty:
        raise ValueError(f"No matching data for {disease_name} - check join keys")

    # Features
    modeling_data['log_pop'] = np.log1p(modeling_data['population'])
    modeling_data['log_pop_std'] = (
        modeling_data['log_pop'] - modeling_data['log_pop'].mean()
    ) / modeling_data['log_pop'].std()

    group_cols = unit_cols + ['year']
    modeling_data['group_id'] = modeling_data.groupby(group_cols).ngroup()
    n_groups = int(modeling_data['group_id'].max()) + 1
    print(f"Fitting Poisson regression over {n_groups:,} aggregation groups...")

    group_ids = modeling_data['group_id'].values
    obs_vals = modeling_data[case_col].values
    logpop_std = modeling_data['log_pop_std'].values
    pop_vals = modeling_data['population'].values

    # Observed total per aggregation group (not summed across all member hexagons)
    group_obs = modeling_data.groupby('group_id')[case_col].first().values

    def neg_log_likelihood(params):
        beta0, beta1 = params
        rate = pop_vals * np.exp(beta0 + beta1 * logpop_std)
        group_pred = np.bincount(group_ids, weights=rate)
        epsilon = 1e-10
        log_lik = np.sum(group_obs * np.log(group_pred + epsilon) - group_pred)
        return -log_lik

    print("Optimizing...")
    result = optimize.minimize(neg_log_likelihood, x0=[0.0, 0.0],
                               method='L-BFGS-B', options={'maxiter': 1000})
    if not result.success:
        warnings.warn(f"Optimization did not converge for {disease_name}: {result.message}")

    beta0, beta1 = result.x
    print(f"Fitted coefficients: b0 = {beta0:.4f}, b1 = {beta1:.4f}")
    if report_file:
        report_file.write(f"Fitted coefficients: b0 = {beta0:.4f}, b1 = {beta1:.4f}\n")
        report_file.write("Population coefficient interpretation: ")
        report_file.write("POSITIVE (urban/high-density concentration)\n" if beta1 > 0
                          else "NEGATIVE (rural/low-density concentration)\n")

    # Mass-preserving rescale (vectorized)
    print("Generating mass-preserving predictions...")
    raw_pred = pop_vals * np.exp(beta0 + beta1 * logpop_std)
    modeling_data['raw_pred'] = raw_pred

    group_raw_sum = modeling_data.groupby('group_id')['raw_pred'].transform('sum')
    group_size = modeling_data.groupby('group_id')['raw_pred'].transform('size')
    scale = np.where(group_raw_sum.values > 0,
                     obs_vals / group_raw_sum.values, 0.0)
    modeling_data['final_pred'] = np.where(
        group_raw_sum.values > 0,
        modeling_data['raw_pred'].values * scale,
        obs_vals / group_size.values
    )

    # Validate mass preservation
    print("Validating mass preservation...")
    validation = (modeling_data.groupby(group_cols)
                  .agg({case_col: 'first', 'final_pred': 'sum'})
                  .reset_index())
    validation['abs_error'] = np.abs(validation[case_col] - validation['final_pred'])
    max_error = validation['abs_error'].max()
    mean_error = validation['abs_error'].mean()
    print(f"Mass preservation check:")
    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  Mean absolute error: {mean_error:.6f}")
    if report_file:
        report_file.write("Mass preservation validation:\n")
        report_file.write(f"  Max absolute error: {max_error:.6f}\n")
        report_file.write(f"  Mean absolute error: {mean_error:.6f}\n")
        report_file.write(f"  Units validated: {len(validation):,}\n\n")
    if max_error > 1e-6:
        warnings.warn(f"Mass preservation error {max_error} exceeds tolerance")

    output_cols = ['h3_index'] + group_cols + ['population', 'final_pred']
    result_df = modeling_data[output_cols].copy()
    result_df = result_df.rename(columns={'final_pred': f'{disease_name}_cases'})
    print(f"Generated {len(result_df):,} hexagon-level predictions")

    del modeling_data, hex_data
    gc.collect()
    return result_df


def main():
    print("VectorHotspot Real Data Disaggregation Pipeline")
    print("=" * 50)
    print("Loading datasets...")

    h3_grid = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'india_h3_grid_res7.csv',
        encoding='utf-8-sig')
    h3_grid.columns = h3_grid.columns.str.strip()
    print(f"H3 grid: {len(h3_grid):,} hexagons")

    # Hexagon population — kept in WIDE format
    hex_pop = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'hex_population_2000_2020.csv',
        encoding='utf-8-sig')
    hex_pop.columns = hex_pop.columns.str.strip()
    n_years = len([c for c in hex_pop.columns if c.startswith('population')])
    print(f"Hexagon population: {len(hex_pop):,} hexagons x {n_years} years")

    # Dengue (state-level): adm_1_name, Year, dengue_total
    dengue_df = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'dengue_state_2010_2024.csv',
        encoding='utf-8-sig')
    dengue_df.columns = dengue_df.columns.str.strip()
    dengue_df = dengue_df.rename(columns={
        'adm_1_name': 'state', 'Year': 'year', 'dengue_total': 'cases'})
    dengue_df = dengue_df[['state', 'year', 'cases']].dropna(subset=['cases', 'state'])
    dengue_df['cases'] = dengue_df['cases'].astype(float)
    dengue_df['state'] = dengue_df['state'].apply(normalize_state)
    print(f"Dengue data: {len(dengue_df):,} state-year rows, "
          f"years {dengue_df['year'].min()}-{dengue_df['year'].max()}")

    # Malaria (district-level): Year, State, District, Total_Cases, Total_Deaths
    malaria_df = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'malaria_district_2000_2024.csv',
        encoding='utf-8-sig')
    malaria_df.columns = malaria_df.columns.str.strip()
    malaria_df = malaria_df.rename(columns={
        'Year': 'year', 'State': 'state', 'District': 'district',
        'Total_Cases': 'cases'})
    malaria_df = malaria_df[['state', 'district', 'year', 'cases']].dropna(subset=['cases'])
    malaria_df['cases'] = malaria_df['cases'].astype(float)
    malaria_df['state'] = malaria_df['state'].apply(normalize_state)
    print(f"Malaria data: {len(malaria_df):,} district-year rows, "
          f"years {malaria_df['year'].min()}-{malaria_df['year'].max()}")

    report_path = PROJECT_ROOT / 'data' / 'processed' / 'disaggregation_fit_report.txt'
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write("VectorHotspot Disaggregation Model Fit Report\n")
        report_file.write(f"Generated: {pd.Timestamp.now()}\n")
        report_file.write("=" * 60 + "\n\n")

        dengue_hex = fit_and_disaggregate(
            case_df=dengue_df, unit_cols=['state'], case_col='cases',
            hex_pop_wide=hex_pop, hex_grid=h3_grid,
            disease_name='dengue', report_file=report_file)
        dengue_out = PROJECT_ROOT / 'data' / 'processed' / 'dengue_hex_annual.csv'
        dengue_hex.to_csv(dengue_out, index=False, encoding='utf-8')
        print(f"Dengue results saved: {dengue_out}")
        print(f"File size: {dengue_out.stat().st_size / (1024*1024):.1f} MB")
        total_dengue = dengue_hex['dengue_cases'].sum()
        n_dengue = len(dengue_hex)
        del dengue_hex; gc.collect()

        malaria_hex = fit_and_disaggregate(
            case_df=malaria_df, unit_cols=['state', 'district'], case_col='cases',
            hex_pop_wide=hex_pop, hex_grid=h3_grid,
            disease_name='malaria', report_file=report_file)
        malaria_out = PROJECT_ROOT / 'data' / 'processed' / 'malaria_hex_annual.csv'
        malaria_hex.to_csv(malaria_out, index=False, encoding='utf-8')
        print(f"Malaria results saved: {malaria_out}")
        print(f"File size: {malaria_out.stat().st_size / (1024*1024):.1f} MB")
        total_malaria = malaria_hex['malaria_cases'].sum()
        n_malaria = len(malaria_hex)
        del malaria_hex; gc.collect()

        print(f"\nSUMMARY:")
        print(f"Total predicted dengue cases: {total_dengue:,.0f}")
        print(f"Total predicted malaria cases: {total_malaria:,.0f}")
        report_file.write("FINAL SUMMARY:\n")
        report_file.write(f"Total predicted dengue cases: {total_dengue:,.0f}\n")
        report_file.write(f"Total predicted malaria cases: {total_malaria:,.0f}\n")
        report_file.write(f"Dengue hexagon-years: {n_dengue:,}\n")
        report_file.write(f"Malaria hexagon-years: {n_malaria:,}\n")

    print(f"\nFit report saved: {report_path}")
    print("\nDisaggregation pipeline completed successfully!")


if __name__ == '__main__':
    main()