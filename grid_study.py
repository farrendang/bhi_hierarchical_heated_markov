import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timedelta
import os
from collections import defaultdict

def fix_data(data):
    fixed_data = data.copy()
    for df in fixed_data:
        df['Exercise duration_s'] /= 100
        temp = df['Sleep type duration_minutes'].copy()
        df['Sleep type duration_minutes'] = df['Sleep duration_minutes']
        df['Sleep duration_minutes'] = temp

    return fixed_data

def augment_data(df):
    # --- Parse datetime ---
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y=%m-%d %H:%M', errors='coerce')
    df = df.sort_values('Datetime').reset_index(drop=True)

    # --- Compute durations (in minutes) ---
    if 'Exercise duration_s' in df.columns and 'Sleep duration_minutes' in df.columns:
        df['Duration_min'] = (
            df['Exercise duration_s'].fillna(0) // 60  # convert seconds to minutes
            + df['Sleep duration_minutes'].fillna(0)
        )
    elif 'Exercise duration_s' in df.columns:
        df['Duration_min'] = df['Exercise duration_s'].fillna(0) // 60
    elif 'Sleep duration_minutes' in df.columns:
        df['Duration_min'] = df['Sleep duration_minutes'].fillna(0)
    else:
        raise ValueError("Expected 'Exercise duration_s' and/or 'Sleep duration_minutes' columns.")
    # --- Build expanded records ---
    augmented_rows = []
    for i, row in df.iterrows():
        current_time = row['Datetime']
        duration = int(row['Duration_min'])
        activity = row['Activity_Type']

        # Compute gap to previous record
        if i > 0:
            prev_time = df.loc[i-1, 'Datetime']
            gap_minutes = int((current_time - prev_time).total_seconds() // 60 - 1)
        else:
            gap_minutes = duration  # no previous record, so just use duration

        # Use whichever is shorter
        expand_minutes = min(duration, gap_minutes)
        if expand_minutes <= 0:
            # No expansion needed, just append the current record
            augmented_rows.append({'Datetime': current_time, 'Activity_Type': activity})
            continue

        # Generate timestamps for each minute before current_time (exclusive)
        times = pd.date_range(
            end=current_time,
            periods=expand_minutes + 1,  # +1 to include current_time
            freq='min'
        )

        # Append one row per minute
        for t in times:
            augmented_rows.append({'Datetime': t, 'Activity_Type': activity})
    # --- Return new DataFrame ---
    augmented_df = pd.DataFrame(augmented_rows)
    return augmented_df.reset_index(drop=True)

def compute_activity_histogram_transition_matrices_target(data, activity_state_dict, max_gap_hours = 8):
    activity_state_dict_inverse = {val:key for key, val in activity_state_dict.items()}
    num_states = len(activity_state_dict.keys())
    max_gap_hours = 8

    #Get target distribution
    activity_histogram = np.ones((24, num_states))
    for df in data:
        # Map the activity strings to numeric codes in one go
        activities = df['Activity_Type'].map(activity_state_dict_inverse)

        # Convert to datetime efficiently (only once)
        times = pd.to_datetime(df['Datetime'], format='%Y=%m-%d %H:%M', errors='coerce')
        hours = times.dt.hour.to_numpy()

        # Use NumPy histogram2d to bin by (hour, activity)
        h, _, _ = np.histogram2d(
            hours,
            activities,
            bins=[np.arange(25), np.arange(len(activity_state_dict_inverse) + 1)]
        )
        # Accumulate into your main histogram
        activity_histogram += h.astype(int)

    target = np.sum(activity_histogram, axis = 0)
    target = target / np.sum(target)

    activity_histogram /= np.sum(activity_histogram, axis = 1, keepdims = True)

    #Augment data
    augmented_data = [augment_data(df) for df in data]

    #Activity distribution, use ones for pseudocounts
    # activity_histogram = np.ones((24, num_states))
    transition_matrices = np.ones((24, num_states, num_states))
    for df in augmented_data:
        times = pd.to_datetime(df['Datetime'], format='%Y=%m-%d %H:%M', errors='coerce')
        hours = times.dt.hour.to_numpy()

        # Map activity types to integer codes
        activities = df['Activity_Type'].map(activity_state_dict_inverse).to_numpy()

        # --- Compute time differences and hours ---
        time_diffs = (times.diff().dt.total_seconds() / 3600.0).to_numpy()
        current_hours = times.dt.hour.to_numpy()

        # --- Iterate through pairs (vectorized indexing is tricky here) ---
        hour = 0
        i = 0
        j = 0
        for t in range(len(df) - 1):
            i = activities[t]
            j = activities[t + 1]
            gap = time_diffs[t + 1]
            if np.isnan(gap) or gap > max_gap_hours:
                continue
            hour = current_hours[t]
            # activity_histogram[hour][i] += 1
            transition_matrices[hour][i][j] += 1
        # activity_histogram[hour][j] += 1

    activity_histogram /= np.sum(activity_histogram, axis = 1, keepdims = True)
    transition_matrices /= np.sum(transition_matrices, axis = 2, keepdims = True)

    return activity_histogram, transition_matrices, target

def compute_cpdfs(data, activity_state_dict):
    activity_state_dict_inverse = {val:key for key, val in activity_state_dict.items()}
    # Initialize accumulators
    combined_hr = defaultdict(lambda: defaultdict(int))
    combined_ad_tier3 = defaultdict(lambda: defaultdict(int))

    tier_3_dependencies = {
        'Sleep duration_minutes': 'Sleep type duration_minutes',
        'Exercise duration_s': 'Calories burned_kcal'
    }

    for df in data:
        working_df = df.copy()
        working_df['activity duration'] = (
            working_df['Sleep duration_minutes'].fillna(0) +
            working_df['Exercise duration_s'].fillna(0)
        )
        working_df['tier 3 var'] = (
            working_df['Calories burned_kcal'].fillna(0) +
            working_df['Sleep type duration_minutes'].fillna(0)
        )

        # --- Heart Rate Histogram per Activity ---
        activity_hr_hist = (
            working_df.groupby('Activity_Type')['Heart rate___beats/minute']
            .value_counts()
        )

        for (activity, hr), count in activity_hr_hist.items():
            combined_hr[activity_state_dict_inverse[activity]][hr] += count

        # --- Activity Duration × Tier 3 Histogram per Activity ---
        # Build tuple keys for joint histograms
        working_df['pair'] = list(zip(working_df['activity duration'], working_df['tier 3 var']))
        activity_ad_tier3_hist = (
            working_df.groupby('Activity_Type')['pair']
            .value_counts()
        )

        for (activity, pair), count in activity_ad_tier3_hist.items():
            combined_ad_tier3[activity_state_dict_inverse[activity]][pair] += count


    # --- Convert to normalized conditional PDFs ---
    def normalize_nested_dict(d):
        normalized = {}
        for key, subdict in d.items():
            total = sum(subdict.values())
            if total > 0:
                normalized[key] = {k: v / total for k, v in subdict.items()}
            else:
                normalized[key] = {}
        return normalized

    cpdfs_hr = normalize_nested_dict(combined_hr)
    cpdfs_ad_tier3 = normalize_nested_dict(combined_ad_tier3)

    return cpdfs_hr, cpdfs_ad_tier3

def matrix_factorize(matrix, p=1):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    real_factorized_matrix = np.real(eigenvectors @ np.diag(np.nan_to_num(np.exp(1/p * np.log(1e-6+eigenvalues)), nan=0.0)) @ np.linalg.inv(eigenvectors)).astype(np.float64)
    real_factorized_matrix = np.maximum(0, np.minimum(1, real_factorized_matrix))
    stochastic_factorized_matrix = real_factorized_matrix / real_factorized_matrix.sum(axis=1, keepdims=True)
    return stochastic_factorized_matrix


def pmf_to_pmf_and_cdf_array_1d(pmf_dict):
    """
    Convert a PMF dict {value: prob_mass} into arrays suitable for sampling.

    Returns:
        values (np.ndarray): sorted unique values
        cdf (np.ndarray): cumulative probabilities in [0, 1]
    """
    # Sort PMF by key
    values, probs = zip(*sorted(pmf_dict.items()))
    values = np.concat([(0,)+values])
    probs = np.concat([(0, )+probs])
    # probs = np.array(probs, dtype=float)

    # Normalize (in case not already normalized)
    probs /= probs.sum()

    # Compute CDF
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0  # Ensure exact 1.0

    return np.array(values), probs, cdf

def sample_from_cdf(values, cdf, n_samples=1):
    """
    Sample from an empirical CDF using linear interpolation.

    Args:
        values (np.ndarray): Sorted 1D array of support values (shape N,).
        cdf (np.ndarray): Corresponding cumulative probabilities in [0, 1] (shape N,).
        n_samples (int): Number of samples to draw.

    Returns:
        samples (np.ndarray): shape (n_samples,)
    """
    if not np.all(np.diff(cdf) >= 0):
        raise ValueError("CDF must be non-decreasing.")
    if cdf[0] > 0 or cdf[-1] < 1:
        raise ValueError("CDF must start near 0 and end near 1.")

    # Generate uniform random numbers
    u = np.random.rand(n_samples)

    # Linear interpolation from uniform samples to value domain
    samples = np.interp(u, cdf, values)

    return samples


def hierarchical_sample_from_pmf(pmf_dict, n_samples=1):
    """
    Hierarchical sampling from a multidimensional discrete PMF.

    Args:
        pmf_dict (dict): {(x1, x2, ..., xD): prob_mass}, any dimensionality.
        n_samples (int): number of samples to draw.

    Returns:
        samples (np.ndarray): shape (n_samples, D)
    """
    # Normalize the joint PMF
    total_prob = sum(pmf_dict.values())
    if total_prob == 0:
        raise ValueError("Total probability mass is zero.")
    pmf = {k: v / total_prob for k, v in pmf_dict.items()}

    # Determine number of dimensions
    first_key = next(iter(pmf))
    D = len(first_key)
    samples = np.zeros((n_samples, D), dtype=object)

    # --- Precompute conditional probability tables ---
    # Level 0: marginal P(x0)
    marginal_0 = defaultdict(float)
    for key, p in pmf.items():
        marginal_0[key[0]] += p

    # For each subsequent variable, precompute conditional PMFs
    cond_tables = [None]  # cond_tables[d] = {prefix: {x_d: P(x_d | prefix)}}
    for d in range(1, D):
        cond_d = defaultdict(lambda: defaultdict(float))
        for key, p in pmf.items():
            prefix = key[:d]
            cond_d[prefix][key[d]] += p
        # Normalize conditionals for each prefix
        for prefix, subpmf in cond_d.items():
            total = sum(subpmf.values())
            for x in subpmf:
                subpmf[x] /= total
        cond_tables.append(cond_d)

    for i in range(n_samples):
        xs = []

        # Sample first variable from marginal
        x0_vals, x0_probs = zip(*sorted(marginal_0.items()))
        x0_vals = np.array(x0_vals, dtype=float)
        x0_cdf = np.cumsum(x0_probs)
        x0_cdf /= x0_cdf[-1]  # normalize just in case

        u = np.random.rand()
        # Linear interpolation of inverse CDF
        x0 = np.interp(u, x0_cdf, x0_vals)
        xs.append(x0)

        # Sequentially sample each conditional variable
        for d in range(1, D):
            prefix = (x0_vals[np.searchsorted(x0_cdf,u)], )
            cond_d = cond_tables[d][prefix]
            vals, probs = zip(*sorted(cond_d.items()))
            vals = np.array(vals, dtype=float)
            cdf = np.cumsum(probs)
            cdf /= cdf[-1]

            u2 = np.random.rand()
            x = np.interp(u2, cdf, vals)
            xs.append(x)

        samples[i, :] = xs

    return np.array(samples, dtype=object)

def tier_1(activity_histogram, transition_matrices, target, start_time, end_time, heat_step_size, warm_up_period):
    """
    activity_histogram: np.array of shape (24, num_states)
    transition_matrices: np.array of shape (24, num_states, num_states)
    target: np.array of shape (num_states,)
    start_time: datetime
    end_time: datetime
    heat_step_size: float
    warm_up_period: int

    Returns synth_activity_sequence: pd.DataFrame of shape (T, 2)
    """
    synth_activity_sequence = []
    K = len(target)
    T = int((end_time - start_time).total_seconds() // 60)
    current_time = start_time
    h = current_time.hour
    activity_probs = activity_histogram[h].copy()
    transition_matrix = matrix_factorize(transition_matrices[h].copy(), p = 60)
    beta = np.ones(K)
    for t in range(T+1):
        current_time = current_time + timedelta(minutes = 1)
        h = current_time.hour
        if t % 60 == 0:
            activity_probs = activity_histogram[h].copy()
            transition_matrix = matrix_factorize(transition_matrices[h].copy(), p = 60)
            beta = np.ones(K)
        else:
            activity_probs = activity_probs @ transition_matrix

        if t > warm_up_period:
            beta = np.maximum(0, np.minimum(1, beta + heat_step_size  * (target - activity_probs)))
            for k in range(K):
                excess = min(1, max(0, transition_matrix[k][k] - beta[k]))
                transition_matrix[k] += excess / (K-1)
                transition_matrix[k][k] = beta[k]
        synth_activity = np.random.choice(K, p = activity_probs / np.sum(activity_probs))
        synth_activity_sequence.append(pd.DataFrame({'Datetime': current_time, 'Activity_Type': synth_activity}, index = [t]))
    synth_activity_sequence = pd.concat(synth_activity_sequence)
    return synth_activity_sequence

def tier_2_3(synth_activity_sequence, cpmfs_hr, cpmfs_ad_tier3, activity_state_dict, chain_len = 1):

    def pmf_variance(values, pmf):
        mean = sum(x * p for x, p in zip(values, pmf))
        mean_sq = sum((x**2) * p for x, p in zip(values, pmf))
        return mean_sq - mean**2

    heart_rates = []
    pmfs_cdfs = {key:[pmf_to_pmf_and_cdf_array_1d(value)] for key, value in cpmfs_hr.items()}

    #Heart rate
    i = 0
    while i < len(synth_activity_sequence):
        activity = synth_activity_sequence['Activity_Type'].iloc[i]
        values, pmf, cdf = pmfs_cdfs[activity][0]
        subchain = []
        subchain.extend(sample_from_cdf(values, cdf))
        prop_variance = pmf_variance(values, pmf)
        for j in range(chain_len):
            current = subchain[-1]
            proposal = np.random.normal(loc = current, scale = np.sqrt(prop_variance))
            p_current = np.interp(current, values, pmf)
            p_proposal = np.interp(proposal, values, pmf)
            alpha = min(1, p_proposal/p_current)
            if np.random.rand() < alpha:
                subchain.append(proposal)
            else:
                subchain.append(current)
        subchain = np.array(subchain)
        heart_rates.extend(list(subchain[1:]))
        i += chain_len

    # Pre-allocate columns with NaN
    ad_tier_3_df = pd.DataFrame({
        'Calories burned_kcal': np.nan,
        'Exercise duration_s': np.nan,
        'Sleep duration_minutes': np.nan,
        'Sleep type duration_minutes': np.nan,
        'Floors climbed___floors': np.nan
    }, index=synth_activity_sequence.index)

    # Handle deterministic cases first
    no_activity_idx = synth_activity_sequence['Activity_Type'].isin([0])
    floors_idx = synth_activity_sequence['Activity_Type'] == 5

    ad_tier_3_df.loc[floors_idx, 'Floors climbed___floors'] = np.float64(1)

    # Group remaining by activity type
    for activity, group_idx in synth_activity_sequence[~synth_activity_sequence['Activity_Type'].isin([0, 5])].groupby('Activity_Type').groups.items():
        cpmf_dict = cpmfs_ad_tier3[activity]

        # Sample once per row in the group
        samples = np.array([hierarchical_sample_from_pmf(cpmf_dict)[0] for _ in range(len(group_idx))]).astype(np.float64)

        if activity in [1, 2]:  # Sleeping
            ad_tier_3_df.loc[group_idx, 'Sleep duration_minutes'] = np.round(samples[:, 0])
            ad_tier_3_df.loc[group_idx, 'Sleep type duration_minutes'] = np.round(samples[:, 1])
        elif activity in [3, 4]:  # Exercise
            ad_tier_3_df.loc[group_idx, 'Exercise duration_s'] = np.round(samples[:, 0], decimals = 2)
            ad_tier_3_df.loc[group_idx, 'Calories burned_kcal'] = np.round(samples[:, 1])
    synth_activity_sequence['Activity_Type'] = synth_activity_sequence['Activity_Type'].map(activity_state_dict)
    return pd.concat([synth_activity_sequence, pd.Series(np.round(heart_rates), name = 'Heart rate___beats/minute'), ad_tier_3_df], axis = 1)

data_dir = 'SyntheticData'
data = fix_data([pd.read_csv(path) for path in glob.glob(data_dir + '/*.csv') if 'User' in path])

activity_state_dict = {
    0: 'No Physical Activity',
    1: 'REM Sleep',
    2: 'Light Sleep',
    3: 'Running',
    4: 'Walking',
    5: 'Floors Climbed'
}

activity_histogram, transition_matrices, target = compute_activity_histogram_transition_matrices_target(data, activity_state_dict, max_gap_hours = 8)
cpmfs_hr, cpmfs_ad_tier3 = compute_cpdfs(data, activity_state_dict)

start_time = datetime(2022, 12, 8, 0, 0)
end_time = datetime(2022, 12, 22, 0, 0)
# heat_step_size = 0.05
# warm_up_period = 2000
heat_step_size_grid = [0, 0.05, 0.1, 0.2]
warm_up_period_grid = [4000, 8000]
parent_dir = '' # put working directory here
for heat_step_size in heat_step_size_grid:
    for warm_up_period in warm_up_period_grid:
        output_dir = os.path.join(parent_dir, f'sim_alpha_{int(heat_step_size*100)}_warmup_{warm_up_period}')
        os.makedirs(output_dir, exist_ok = True)
        for i in range(0, 100):
            output_filepath = os.path.join(output_dir, f'SynthUser{i}.csv')
            if not os.path.exists(output_filepath):
                datetime_activity = tier_1(activity_histogram, transition_matrices, target, start_time, end_time, heat_step_size, warm_up_period)
                synth_record_full = tier_2_3(datetime_activity, cpmfs_hr, cpmfs_ad_tier3, activity_state_dict)
                synth_record_full.to_csv(output_filepath)
            else:
                print(f'File {output_filepath} already exists. Skipping.')


