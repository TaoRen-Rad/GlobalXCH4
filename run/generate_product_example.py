import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xch4_retrieval.process_data import standardize_data
from xch4_retrieval.train import load_model
from xch4_retrieval.model import ensemble_predict


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(script_dir, "20240101T074458_20240101T092629.parquet")
    model_path = os.path.join(script_dir, "xch4_finetuned_model.pth")
    scalers_path = os.path.join(script_dir, "scalers_combined.parquet")
    output_file = os.path.join(script_dir, "20240101T074458_20240101T092629_product.parquet")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    checkpoint = load_model(model_path, device)
    snapshots = checkpoint['snapshots']
    output_mean = checkpoint['output_mean']
    output_scale = checkpoint['output_scale']
    feature_columns = checkpoint['feature_columns']
    
    print("Reading input data...")
    df = pd.read_parquet(input_file)
    
    print("Standardizing data...")
    temp_standardized = os.path.join(script_dir, "temp_standardized.parquet")
    df_standardized = standardize_data(df, scalers_path, temp_standardized)
    
    print("Preparing features...")
    X = torch.tensor(df_standardized[feature_columns].values, dtype=torch.float32).to(device)
    
    print("Making predictions...")
    ensemble_mean, ensemble_variance = ensemble_predict(snapshots, X, device)
    
    methane_mixing_ratio = ensemble_mean.flatten() * output_scale + output_mean
    uncertainty = np.sqrt(ensemble_variance.flatten()) * output_scale
    
    print("Creating product file...")
    product_df = pd.DataFrame({
        "longitude": df["longitude"].values,
        "latitude": df["latitude"].values,
        "scanline": df["scanline"].values,
        "pixel": df["pixel"].values,
        "methane_mixing_ratio": methane_mixing_ratio,
        "uncertainty": uncertainty
    })
    
    product_df.to_parquet(output_file, index=False)
    print(f"Product saved to: {output_file}")
    print(f"Total records: {len(product_df)}")
    
    os.remove(temp_standardized)


if __name__ == "__main__":
    main()

