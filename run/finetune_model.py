"""
Example script for fine-tuning the CH4 inversion model on GOSAT data.
"""

import os
import sys
import torch
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xch4_retrieval import (
    finetune_model,
    save_model,
    evaluate_model,
    load_model
)


def main():
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_model_path = os.path.join(script_dir, "xch4_model.pth")
    scalers_path = os.path.join(script_dir, "scalers_combined.parquet")
    
    # GOSAT data paths (modify as needed)
    base_dir = "/home/luoyuwen/workspace/CH4/reverse/03_final_train/06_fine_file"
    gosat_data_paths = [
        os.path.join(base_dir, "gt_2019.parquet"),
        os.path.join(base_dir, "gt_2020.parquet"),
        os.path.join(base_dir, "gt_2021.parquet"),
        os.path.join(base_dir, "gt_2022.parquet"),
        os.path.join(base_dir, "gt_2023.parquet"),
    ]
    
    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gosat_dfs = []
    for path in gosat_data_paths:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if 'year' not in df.columns:
                year = int(os.path.basename(path).split('_')[1][:4])
                df['year'] = year
            gosat_dfs.append(df)
    
    if not gosat_dfs:
        raise ValueError("No GOSAT data files found")
    
    gosat_df = pd.concat(gosat_dfs, ignore_index=True)
    gosat_df = gosat_df[gosat_df['year'].isin([2019, 2020, 2021, 2022, 2023])]
    
    train_list = []
    val_list = []
    for year in range(2019, 2024):
        year_data = gosat_df[gosat_df['year'] == year]
        year_data = year_data.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(0.8 * len(year_data))
        train_list.append(year_data[:split_idx])
        val_list.append(year_data[split_idx:])
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    
    feature_columns = [col for col in train_df.columns 
                      if col not in ["gosat_xch4", "longitude", "latitude", "year"]]
    
    X_train = torch.tensor(train_df[feature_columns].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_df["gosat_xch4"].values.reshape(-1, 1), dtype=torch.float32).to(device)
    X_val = torch.tensor(val_df[feature_columns].values, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_df["gosat_xch4"].values.reshape(-1, 1), dtype=torch.float32).to(device)
    
    scalers_df = pd.read_parquet(scalers_path)
    output_scaler = scalers_df[(scalers_df['parameter_group'] == 'xch4') & 
                               (scalers_df['parameter_name'] == 'xch4')]
    output_mean = output_scaler['mean'].values[0]
    output_scale = output_scaler['scale'].values[0]
    
    model, snapshots, best_epoch, best_val_loss = finetune_model(
        pretrained_model_path=pretrained_model_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        num_epochs=65,
        batch_size=512,
        learning_rate=0.0004
    )
    
    finetuned_model_path = os.path.abspath(os.path.join(output_dir, "xch4_finetuned_model.pth"))
    save_model(
        snapshots, best_epoch, best_val_loss,
        output_mean, output_scale, feature_columns,
        finetuned_model_path
    )
    
    checkpoint = load_model(pretrained_model_path, device)
    evaluate_model(
        checkpoint['snapshots'], X_val, y_val, device,
        output_mean, output_scale,
        dataset_name="Validation Set (Original)"
    )
    
    evaluate_model(
        snapshots, X_val, y_val, device,
        output_mean, output_scale,
        dataset_name="Validation Set (Fine-tuned)"
    )
    
    # Test 2024 data
    test_2024_path = os.path.join(base_dir, "gt_2024.parquet")
    if os.path.exists(test_2024_path):
        test_df_2024 = pd.read_parquet(test_2024_path)
        if 'year' not in test_df_2024.columns:
            test_df_2024['year'] = 2024
        
        X_test_2024 = torch.tensor(test_df_2024[feature_columns].values, dtype=torch.float32).to(device)
        y_test_2024 = torch.tensor(test_df_2024["gosat_xch4"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        
        evaluate_model(
            snapshots, X_test_2024, y_test_2024, device,
            output_mean, output_scale,
            dataset_name="Test Set 2024 (Fine-tuned)"
        )
    
    # Test 2025 data
    test_2025_path = os.path.join(base_dir, "gt_2025.parquet")
    if os.path.exists(test_2025_path):
        test_df_2025 = pd.read_parquet(test_2025_path)
        if 'year' not in test_df_2025.columns:
            test_df_2025['year'] = 2025
        
        X_test_2025 = torch.tensor(test_df_2025[feature_columns].values, dtype=torch.float32).to(device)
        y_test_2025 = torch.tensor(test_df_2025["gosat_xch4"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        
        evaluate_model(
            snapshots, X_test_2025, y_test_2025, device,
            output_mean, output_scale,
            dataset_name="Test Set 2025 (Fine-tuned)"
        )
    
    print(f"Fine-tuning completed, model saved to: {finetuned_model_path}")


if __name__ == "__main__":
    main()

