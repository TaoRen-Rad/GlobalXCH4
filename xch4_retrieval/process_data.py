import ast
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .split_data import split_by_time_sequence


ANGLE_COLUMNS = ['solar_zenith_angle', 'solar_azimuth_angle', 'viewing_zenith_angle', 'viewing_azimuth_angle']
META_COLUMNS = ['longitude', 'latitude', 'year', 'time_sequence']
RADIANCE_COLUMNS = [f'radiance_{i}' for i in range(480)]
PROFILE_COLUMNS = [f'methane_profile_{i}' for i in range(12)]
ONE_DIM_COLUMNS = ['surface_pressure', 'earth_sun_distance'] + [
    f'{angle}_{function}' for angle in ANGLE_COLUMNS for function in ('sin', 'cos')
]
FEATURE_COLUMNS = RADIANCE_COLUMNS + PROFILE_COLUMNS + ONE_DIM_COLUMNS


def _parse_array(value, length):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"Expected {length} finite values")
    return array


def _preprocess_data(df, strict=False):
    """Expand spectra and profiles and transform observation geometry."""
    df = df.reset_index(drop=True).copy()
    if df.empty:
        raise ValueError("Input data is empty")
    radiance = np.vstack([_parse_array(value, 480) for value in df['radiance']])
    methane = np.vstack([_parse_array(value, 12) for value in df['methane_profile_apriori']])
    wavelengths = np.linspace(2300, 2343, 480)
    reference = radiance[:, (wavelengths >= 2313) & (wavelengths <= 2313.5)].mean(axis=1, keepdims=True)
    if strict and np.any(reference <= 0):
        raise ValueError("Radiance reference must be positive")
    radiance = np.divide(radiance, reference, out=np.zeros_like(radiance), where=reference > 0)
    for column in ANGLE_COLUMNS:
        radians = np.radians(df[column].to_numpy(dtype=float))
        df[f'{column}_sin'] = np.sin(radians)
        df[f'{column}_cos'] = np.cos(radians)
    if 'year' not in df and 'time_sequence' in df:
        df['year'] = df['time_sequence'].str[:4].astype(int)
    columns = ONE_DIM_COLUMNS + [column for column in ['xch4', *META_COLUMNS] if column in df]
    result = pd.concat([
        pd.DataFrame(radiance, columns=RADIANCE_COLUMNS),
        pd.DataFrame(methane, columns=PROFILE_COLUMNS),
        df[columns],
    ], axis=1)
    numeric = [column for column in result if column not in META_COLUMNS]
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Features or labels contain NaN or Inf")
    return result


def fit_scalers(train_df, output_path):
    """Fit standardization parameters on training features and labels."""
    rows = []
    for group, columns in (
        ('radiance', RADIANCE_COLUMNS), ('methane_profile', PROFILE_COLUMNS),
        ('one_dim', ONE_DIM_COLUMNS), ('xch4', ['xch4']),
    ):
        values = train_df[columns].to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Scaler training data must be nonempty and finite")
        scaler = StandardScaler().fit(values)
        rows.extend(
            {'parameter_group': group, 'parameter_name': column, 'mean': mean, 'scale': scale}
            for column, mean, scale in zip(columns, scaler.mean_, scaler.scale_)
        )
    result = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def generate_scalers_from_raw_data(data_paths, output_path, years):
    frames = []
    for path in data_paths:
        frame = pd.read_parquet(path)
        if 'year' not in frame:
            frame['year'] = int(re.search(r'\d{4}', Path(path).name).group())
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    train = []
    for year in years:
        subset, _ = split_by_time_sequence(combined.loc[combined['year'] == year], seed=42 + year - min(years))
        train.append(_preprocess_data(subset))
    return fit_scalers(pd.concat(train, ignore_index=True), output_path)


def load_scalers(scalers_path):
    table = pd.read_parquet(scalers_path)
    if table['parameter_name'].duplicated().any():
        raise ValueError("Duplicate scaler parameters")
    table = table.set_index('parameter_name')
    if not np.isfinite(table[['mean', 'scale']].to_numpy()).all() or (table['scale'] <= 0).any():
        raise ValueError("Scaler parameters must be finite with positive scales")
    return table


def standardize_data(df, scalers_path, output_path=None, preprocessed_df=None, strict=False):
    scalers = load_scalers(scalers_path)
    result = _preprocess_data(df, strict=strict) if preprocessed_df is None else preprocessed_df.copy()
    columns = [column for column in result if column not in META_COLUMNS]
    scaled = (result[columns] - scalers.loc[columns, 'mean']) / scalers.loc[columns, 'scale']
    result = pd.concat([scaled, result.drop(columns=columns)], axis=1)[result.columns]
    if not np.isfinite(result[columns].to_numpy(dtype=float)).all():
        raise ValueError("Standardized data contains NaN or Inf")
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
    return result


def filter_gosat_matches(df, max_distance_km=10.0):
    """Select TROPOMI–GOSAT pairs by Haversine distance."""
    coordinates = ['latitude', 'longitude', 'gosat_latitude', 'gosat_longitude']
    scalars = coordinates + ['xch4', 'gosat_xch4', 'surface_pressure', 'earth_sun_distance', *ANGLE_COLUMNS]
    required = scalars + ['time_sequence', 'radiance', 'methane_profile_apriori']
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"Missing matched-data columns: {sorted(missing)}")
    result = df.dropna(subset=required).copy()
    result = result.loc[np.isfinite(result[scalars].to_numpy(dtype=float)).all(axis=1)].copy()
    valid = (result['latitude'].abs() <= 90) & (result['gosat_latitude'].abs() <= 90)
    valid &= (result['longitude'].abs() <= 180) & (result['gosat_longitude'].abs() <= 180)
    result = result.loc[valid].copy()
    lat1, lon1, lat2, lon2 = np.radians(result[coordinates].to_numpy(dtype=float)).T
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    a = np.clip(a, 0, 1)
    result['distance_km'] = 2 * 6371.0088 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    result = result.loc[result['distance_km'] <= max_distance_km].copy()
    for column, length in (('radiance', 480), ('methane_profile_apriori', 12)):
        def valid_array(value):
            try:
                _parse_array(value, length)
                return True
            except (TypeError, ValueError, SyntaxError):
                return False
        result = result.loc[result[column].map(valid_array)].copy()
    if result.empty:
        raise ValueError("No valid TROPOMI–GOSAT pairs within the distance limit")
    return result.reset_index(drop=True)


def standardize_gosat_data(df, scalers_path):
    result = standardize_data(df, scalers_path)
    scaler = load_scalers(scalers_path).loc['xch4']
    result['gosat_xch4'] = df['gosat_xch4'].to_numpy(dtype=float)
    result['gosat_xch4_standardized'] = (result['gosat_xch4'] - scaler['mean']) / scaler['scale']
    result['distance_km'] = df['distance_km'].to_numpy(dtype=float)
    if not np.isfinite(result['gosat_xch4_standardized']).all():
        raise ValueError("GOSAT labels contain NaN or Inf")
    return result
