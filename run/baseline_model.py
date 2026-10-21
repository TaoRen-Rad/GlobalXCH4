import os
import sys
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xch4_retrieval.train import train_model, save_model
from xch4_retrieval.evaluate import evaluate_model
from xch4_retrieval.process_data import FEATURE_COLUMNS


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(output_dir, "train_2019_2023.parquet")
    val_file = os.path.join(output_dir, "val_2019_2023.parquet")
    test_file_2024 = os.path.join(output_dir, "2024_10w.parquet")
    test_file_2025 = os.path.join(output_dir, "2025_10w.parquet")
    scalers_file = os.path.join(output_dir, "scalers_combined.parquet")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading training data...")
    train_df_split = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    if set(train_df_split['time_sequence']) & set(val_df['time_sequence']):
        raise ValueError("Training and validation share overpasses")
    print(f"Total training samples: {len(train_df_split)}")
    print(f"Total validation samples: {len(val_df)}")
    
    feature_columns = list(FEATURE_COLUMNS)
    print(f"Number of features: {len(feature_columns)}")
    
    print("Preparing tensors...")
    X_train = torch.tensor(train_df_split[feature_columns].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_df_split["xch4"].values.reshape(-1, 1), dtype=torch.float32).to(device)
    X_val = torch.tensor(val_df[feature_columns].values, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_df["xch4"].values.reshape(-1, 1), dtype=torch.float32).to(device)
    print(f"Training tensor shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation tensor shape: {X_val.shape}, {y_val.shape}")
    
    print("Loading scalers...")
    scalers_df = pd.read_parquet(scalers_file)
    output_scaler = scalers_df[(scalers_df['parameter_group'] == 'xch4') & 
                               (scalers_df['parameter_name'] == 'xch4')]
    output_mean = output_scaler['mean'].values[0]
    output_scale = output_scaler['scale'].values[0]
    print(f"Output mean: {output_mean:.4f}, Output scale: {output_scale:.4f}")
    
    print("\n" + "="*60)
    print("Starting model training...")
    print("="*60)
    model, snapshots, validation_losses, snapshot_epochs = train_model(
        X_train, y_train, X_val, y_val, device,
        num_epochs=200,
        batch_size=4096,
        learning_rate=0.005
    )
    
    print("\n" + "="*60)
    print("Training Summary:")
    print(f"  Snapshot epochs: {snapshot_epochs}")
    print(f"  Final validation loss: {validation_losses[-1]:.6f}")
    print("="*60)
    
    model_path = os.path.join(output_dir, "xch4_model.pth")
    save_model(
        snapshots, validation_losses, snapshot_epochs,
        output_mean, output_scale, feature_columns,
        model_path
    )
    
    print("\n" + "="*60)
    print("Model Evaluation:")
    print("="*60)
    
    evaluate_model(
        snapshots, X_val, y_val, device,
        output_mean, output_scale,
        dataset_name="Validation Set"
    )
    
    print("\nLoading test data 2024...")
    test_df_2024 = pd.read_parquet(test_file_2024)
    print(f"Loaded {len(test_df_2024)} samples")
    X_test_2024 = torch.tensor(test_df_2024[feature_columns].values, dtype=torch.float32).to(device)
    y_test_2024 = torch.tensor(test_df_2024["xch4"].values.reshape(-1, 1), dtype=torch.float32).to(device)
    
    evaluate_model(
        snapshots, X_test_2024, y_test_2024, device,
        output_mean, output_scale,
        dataset_name="Test Set 2024"
    )
    
    print("\nLoading test data 2025...")
    test_df_2025 = pd.read_parquet(test_file_2025)
    print(f"Loaded {len(test_df_2025)} samples")
    X_test_2025 = torch.tensor(test_df_2025[feature_columns].values, dtype=torch.float32).to(device)
    y_test_2025 = torch.tensor(test_df_2025["xch4"].values.reshape(-1, 1), dtype=torch.float32).to(device)
    
    evaluate_model(
        snapshots, X_test_2025, y_test_2025, device,
        output_mean, output_scale,
        dataset_name="Test Set 2025"
    )
    
    print("\n" + "="*60)
    print(f"Training completed. Model saved to: {model_path}")
    print("="*60)


if __name__ == "__main__":
    main()
