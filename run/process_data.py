import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xch4_retrieval.process_data import _preprocess_data, standardize_data


def main():
    base_dir = "/home/luoyuwen/workspace/CH4/reverse/03_final_train/01_train_file"
    output_dir = os.getcwd()
    
    scalers_path = f"{output_dir}/scalers_combined.parquet"
    
    print("Preprocessing 2019-2023 data...")
    preprocessed_dfs = {}
    for year in range(2019, 2024):
        df = pd.read_parquet(f"{base_dir}/land_{year}_18w.parquet")
        df['year'] = year
        preprocessed_dfs[year] = _preprocess_data(df)
    
    combined_preprocessed = pd.concat(list(preprocessed_dfs.values()), ignore_index=True)
    
    print("Generating scalers...")
    radiance_cols = [f"radiance_{i}" for i in range(480)]
    methane_cols = [f"methane_profile_{i}" for i in range(12)]
    one_dim_cols = ["surface_pressure", "earth_sun_distance",
                    "solar_zenith_angle_sin", "solar_zenith_angle_cos",
                    "solar_azimuth_angle_sin", "solar_azimuth_angle_cos",
                    "viewing_zenith_angle_sin", "viewing_zenith_angle_cos",
                    "viewing_azimuth_angle_sin", "viewing_azimuth_angle_cos"]
    
    scalers_data = []
    
    scaler = StandardScaler()
    scaler.fit(combined_preprocessed[radiance_cols].values)
    for col, mean, scale in zip(radiance_cols, scaler.mean_, scaler.scale_):
        scalers_data.append({"parameter_group": "radiance", "parameter_name": col, "mean": mean, "scale": scale})
    
    scaler = StandardScaler()
    scaler.fit(combined_preprocessed[methane_cols].values)
    for col, mean, scale in zip(methane_cols, scaler.mean_, scaler.scale_):
        scalers_data.append({"parameter_group": "methane_profile", "parameter_name": col, "mean": mean, "scale": scale})
    
    scaler = StandardScaler()
    scaler.fit(combined_preprocessed[one_dim_cols].values)
    for col, mean, scale in zip(one_dim_cols, scaler.mean_, scaler.scale_):
        scalers_data.append({"parameter_group": "one_dim", "parameter_name": col, "mean": mean, "scale": scale})
    
    scaler = StandardScaler()
    scaler.fit(combined_preprocessed[["xch4"]].values)
    scalers_data.append({"parameter_group": "xch4", "parameter_name": "xch4", 
                       "mean": scaler.mean_[0], "scale": scaler.scale_[0]})
    
    scalers_df = pd.DataFrame(scalers_data)
    scalers_df.to_parquet(scalers_path)
    
    print("Standardizing 2019-2023 data...")
    scalers_dict = {row['parameter_name']: {'mean': row['mean'], 'scale': row['scale']} 
                   for _, row in scalers_df.iterrows()}
    
    train_dfs = []
    for year in range(2019, 2024):
        df_scaled = preprocessed_dfs[year].copy()
        for col in df_scaled.columns:
            if col not in ["longitude", "latitude", "year", "time_sequence"]:
                mean = scalers_dict[col]['mean']
                scale = scalers_dict[col]['scale']
                df_scaled[col] = (df_scaled[col] - mean) / scale
        train_dfs.append(df_scaled)
    
    del preprocessed_dfs
    pd.concat(train_dfs, ignore_index=True).to_parquet(f"{output_dir}/2019_2023_18w.parquet", index=False)
    print(f"Saved: {output_dir}/2019_2023_18w.parquet")
    
    print("Standardizing 2024 data...")
    df_2024 = pd.read_parquet(f"{base_dir}/land_2024_10w.parquet")
    df_2024['year'] = 2024
    standardize_data(df_2024, scalers_path, output_path=f"{output_dir}/2024_10w.parquet")
    print(f"Saved: {output_dir}/2024_10w.parquet")
    
    print("Standardizing 2025 data...")
    df_2025 = pd.read_parquet(f"{base_dir}/land_2025_10w.parquet")
    df_2025['year'] = 2025
    standardize_data(df_2025, scalers_path, output_path=f"{output_dir}/2025_10w.parquet")
    print(f"Saved: {output_dir}/2025_10w.parquet")
    

if __name__ == "__main__":
    main()
