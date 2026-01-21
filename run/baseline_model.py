import os
import sys
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xch4_retrieval.train import train_model, save_model
from xch4_retrieval.evaluate import evaluate_model


def main():
    train_file = "/home/luoyuwen/workspace/CH4/reverse/05/run/2019_2023_18w.parquet"
    test_file_2024 = "/home/luoyuwen/workspace/CH4/reverse/05/run/2024_10w.parquet"
    test_file_2025 = "/home/luoyuwen/workspace/CH4/reverse/05/run/2025_10w.parquet"
    scalers_file = "/home/luoyuwen/workspace/CH4/reverse/05/run/scalers_combined.parquet"
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading training data...")
    train_df = pd.read_parquet(train_file)
    print(f"Loaded {len(train_df)} samples from training file")
    
    train_df = train_df.sample(n=900000, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(train_df)} samples for training")
    
    print("Splitting data by year...")
    train_list = []
    val_list = []
    for year in range(2019, 2024):
        year_data = train_df[train_df['year'] == year]
        year_data = year_data.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(0.8 * len(year_data))
        train_list.append(year_data[:split_idx])
        val_list.append(year_data[split_idx:])
        print(f"  Year {year}: {len(year_data[:split_idx])} train, {len(year_data[split_idx:])} val")
    
    train_df_split = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    print(f"Total training samples: {len(train_df_split)}")
    print(f"Total validation samples: {len(val_df)}")
    
    feature_columns = [col for col in train_df_split.columns 
                      if col not in ["xch4", "longitude", "latitude", "year", "time_sequence"]]
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
    model, snapshots, best_epoch, best_val_loss = train_model(
        X_train, y_train, X_val, y_val, device,
        num_epochs=200,
        batch_size=4096,
        learning_rate=0.005
    )
    
    print("\n" + "="*60)
    print("Training Summary:")
    print(f"  Best epoch: {best_epoch + 1}")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print("="*60)
    
    model_path = os.path.join(output_dir, "xch4_model.pth")
    save_model(
        snapshots, best_epoch, best_val_loss,
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
