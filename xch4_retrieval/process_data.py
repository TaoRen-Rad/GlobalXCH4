import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler
from typing import List, Optional


def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data: convert angles, parse arrays, normalize radiance."""
    angle_cols = ["solar_zenith_angle", "solar_azimuth_angle", "viewing_zenith_angle", "viewing_azimuth_angle"]
    for col in angle_cols:
        df[f"{col}_sin"] = np.sin(np.radians(df[col]))
        df[f"{col}_cos"] = np.cos(np.radians(df[col]))
    
    df["radiance"] = df["radiance"].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
    )
    df["methane_profile_apriori"] = df["methane_profile_apriori"].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
    )
    
    radiance_cols = [f"radiance_{i}" for i in range(480)]
    wavelengths = np.linspace(2300, 2343, 480)
    ref_indices = np.where((wavelengths >= 2313) & (wavelengths <= 2313.5))[0]
    radiance = np.vstack(df["radiance"])
    radiance_normalized = np.zeros_like(radiance)
    for i in range(len(radiance)):
        ref_val = np.mean(radiance[i][ref_indices])
        radiance_normalized[i] = radiance[i] / ref_val
    radiance_df = pd.DataFrame(radiance_normalized, columns=radiance_cols)
    
    methane_cols = [f"methane_profile_{i}" for i in range(12)]
    methane_profile = np.vstack(df["methane_profile_apriori"])
    methane_df = pd.DataFrame(methane_profile, columns=methane_cols)
    
    one_dim_cols = ["surface_pressure", "earth_sun_distance",
                    "solar_zenith_angle_sin", "solar_zenith_angle_cos",
                    "solar_azimuth_angle_sin", "solar_azimuth_angle_cos",
                    "viewing_zenith_angle_sin", "viewing_zenith_angle_cos",
                    "viewing_azimuth_angle_sin", "viewing_azimuth_angle_cos"]
    one_dim_df = df[one_dim_cols].copy()
    
    if "year" not in df.columns and "time_sequence" in df.columns:
        df["year"] = df["time_sequence"].str[:4].astype(int)
    
    meta_cols = ["longitude", "latitude", "year", "time_sequence"]
    meta_cols = [col for col in meta_cols if col in df.columns]
    meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame()
    
    feature_dfs = [radiance_df, methane_df, one_dim_df]
    if "xch4" in df.columns:
        feature_dfs.append(df[["xch4"]].copy())
    
    if not meta_df.empty:
        return pd.concat(feature_dfs + [meta_df], axis=1)
    else:
        return pd.concat(feature_dfs, axis=1)


def generate_scalers_from_raw_data(data_paths: List[str], output_path: str, years: List[int]) -> pd.DataFrame:
    """Generate standardization parameters from raw training data."""
    dfs = []
    for path in data_paths:
        df = pd.read_parquet(path)
        if "year" not in df.columns:
            year_match = re.search(r'(\d{4})', os.path.basename(path))
            if year_match:
                df['year'] = int(year_match.group(1))
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df[combined_df["year"].isin(years)]
    
    preprocessed_df = _preprocess_data(combined_df)
    
    radiance_cols = [f"radiance_{i}" for i in range(480)]
    methane_cols = [f"methane_profile_{i}" for i in range(12)]
    one_dim_cols = ["surface_pressure", "earth_sun_distance",
                    "solar_zenith_angle_sin", "solar_zenith_angle_cos",
                    "solar_azimuth_angle_sin", "solar_azimuth_angle_cos",
                    "viewing_zenith_angle_sin", "viewing_zenith_angle_cos",
                    "viewing_azimuth_angle_sin", "viewing_azimuth_angle_cos"]
    
    scalers_data = []
    
    scaler = StandardScaler()
    scaler.fit(preprocessed_df[radiance_cols].values)
    for col, mean, scale in zip(radiance_cols, scaler.mean_, scaler.scale_):
        scalers_data.append({"parameter_group": "radiance", "parameter_name": col, "mean": mean, "scale": scale})
    
    scaler = StandardScaler()
    scaler.fit(preprocessed_df[methane_cols].values)
    for col, mean, scale in zip(methane_cols, scaler.mean_, scaler.scale_):
        scalers_data.append({"parameter_group": "methane_profile", "parameter_name": col, "mean": mean, "scale": scale})
    
    scaler = StandardScaler()
    scaler.fit(preprocessed_df[one_dim_cols].values)
    for col, mean, scale in zip(one_dim_cols, scaler.mean_, scaler.scale_):
        scalers_data.append({"parameter_group": "one_dim", "parameter_name": col, "mean": mean, "scale": scale})
    
    scaler = StandardScaler()
    scaler.fit(preprocessed_df[["xch4"]].values)
    scalers_data.append({"parameter_group": "xch4", "parameter_name": "xch4", 
                       "mean": scaler.mean_[0], "scale": scaler.scale_[0]})
    
    scalers_df = pd.DataFrame(scalers_data)
    scalers_df.to_parquet(output_path)
    
    return scalers_df


def standardize_data(df: Optional[pd.DataFrame], scalers_path: str, output_path: str, preprocessed_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Standardize new data using pre-computed scalers."""
    scalers_df = pd.read_parquet(scalers_path)
    scalers_dict = {row['parameter_name']: {'mean': row['mean'], 'scale': row['scale']} 
                   for _, row in scalers_df.iterrows()}
    
    if preprocessed_df is None:
        preprocessed_df = _preprocess_data(df)
    
    df_scaled = preprocessed_df.copy()
    for col in df_scaled.columns:
        if col not in ["longitude", "latitude", "year", "time_sequence"]:
            mean = scalers_dict[col]['mean']
            scale = scalers_dict[col]['scale']
            df_scaled[col] = (df_scaled[col] - mean) / scale
    
    df_scaled.to_parquet(output_path)
    
    return df_scaled
