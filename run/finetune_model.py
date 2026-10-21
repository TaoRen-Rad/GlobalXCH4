from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xch4_retrieval import finetune_model, save_model, evaluate_model, load_model
from xch4_retrieval.process_data import filter_gosat_matches, standardize_gosat_data, load_scalers
from xch4_retrieval.split_data import split_by_time_sequence


def main():
    script_dir = Path(__file__).resolve().parent
    pretrained_model_path = script_dir / 'xch4_model.pth'
    scalers_path = script_dir / 'scalers_combined.parquet'
    base_dir = script_dir.parent / 'data' / 'tropomi-gosat'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = load_model(pretrained_model_path, torch.device('cpu'))
    feature_columns = checkpoint['feature_columns']
    output_mean = checkpoint['output_mean']
    output_scale = checkpoint['output_scale']
    scaler = load_scalers(scalers_path).loc['xch4']
    if not np.allclose([scaler['mean'], scaler['scale']], [output_mean, output_scale], rtol=1e-10, atol=1e-10):
        raise ValueError('Scaler does not match the pretrained model')

    train_parts, val_parts, test_data = [], [], {}
    for year in range(2019, 2026):
        raw = pd.read_parquet(base_dir / f'gt{year}_0.5_l12.parquet')
        matched = filter_gosat_matches(raw, max_distance_km=10.0)
        matched['year'] = year
        print(f'{year}: {len(matched)} pairs within 10 km')
        if year <= 2023:
            train, val = split_by_time_sequence(matched, seed=42 + year, spatial=False)
            train_parts.append(standardize_gosat_data(train, scalers_path))
            val_parts.append(standardize_gosat_data(val, scalers_path))
        else:
            test_data[year] = standardize_gosat_data(matched, scalers_path)
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    if set(train_df['time_sequence']) & set(val_df['time_sequence']):
        raise ValueError('Training and validation share overpasses')

    def tensors(frame):
        x = frame[feature_columns].to_numpy(dtype=np.float32)
        y = frame['gosat_xch4_standardized'].to_numpy(dtype=np.float32).reshape(-1, 1)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Model inputs or labels contain NaN or Inf')
        return torch.tensor(x, device=device), torch.tensor(y, device=device)

    X_train, y_train = tensors(train_df)
    X_val, y_val = tensors(val_df)
    model, snapshots, validation_losses, snapshot_epochs = finetune_model(
        pretrained_model_path, X_train, y_train, X_val, y_val, device,
        num_epochs=60, batch_size=512, learning_rate=0.0004,
    )
    model_path = script_dir / 'xch4_finetuned_model.pth'
    save_model(
        snapshots, validation_losses, snapshot_epochs,
        output_mean, output_scale, feature_columns, model_path,
    )
    for name, states in (('Original', checkpoint['snapshots']), ('Fine-tuned', snapshots)):
        evaluate_model(states, X_val, y_val, device, output_mean, output_scale, dataset_name=f'Validation ({name})')
    for year, frame in test_data.items():
        x, y = tensors(frame)
        evaluate_model(snapshots, x, y, device, output_mean, output_scale, dataset_name=f'Test {year}')


if __name__ == '__main__':
    main()
