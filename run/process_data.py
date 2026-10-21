from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xch4_retrieval.process_data import _preprocess_data, fit_scalers, standardize_data
from xch4_retrieval.split_data import split_by_time_sequence


def main():
    output_dir = Path(__file__).resolve().parent
    base_dir = output_dir.parent / 'data' / 'training'
    scalers_path = output_dir / 'scalers_combined.parquet'
    train_parts, val_parts = [], []
    for year in range(2019, 2024):
        frame = pd.read_parquet(base_dir / f'land_{year}_18w.parquet')
        frame['year'] = year
        train, val = split_by_time_sequence(frame, seed=42 + year - 2019)
        train_parts.append(_preprocess_data(train))
        val_parts.append(_preprocess_data(val))
        print(f'{year}: {len(train)} train, {len(val)} validation')
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    fit_scalers(train, scalers_path)
    for name, frame in (('train', train), ('val', val)):
        standardize_data(None, scalers_path, output_dir / f'{name}_2019_2023.parquet', preprocessed_df=frame)
    for year in (2024, 2025):
        frame = pd.read_parquet(base_dir / f'land_{year}_10w.parquet')
        frame['year'] = year
        standardize_data(frame, scalers_path, output_dir / f'{year}_10w.parquet')


if __name__ == '__main__':
    main()
