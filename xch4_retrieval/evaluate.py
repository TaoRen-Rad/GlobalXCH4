import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Optional

from .model import ensemble_predict


def evaluate_model(
    models: List[Dict],
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    output_mean: float,
    output_scale: float,
    dataset_name: str = "Dataset",
    verbose: bool = True
) -> Dict:
    """Evaluate model performance on a dataset."""
    ensemble_mean, ensemble_var = ensemble_predict(models, X, device)
    
    ensemble_mean_raw = ensemble_mean.flatten() * output_scale + output_mean
    y_raw = y.cpu().numpy().flatten() * output_scale + output_mean
    ensemble_std_raw = np.sqrt(ensemble_var.flatten()) * output_scale
    
    rmse = np.sqrt(mean_squared_error(y_raw, ensemble_mean_raw))
    r2 = r2_score(y_raw, ensemble_mean_raw)
    me = np.mean(ensemble_mean_raw - y_raw)
    mae = np.mean(np.abs(ensemble_mean_raw - y_raw))
    within_3sigma = np.mean(np.abs(y_raw - ensemble_mean_raw) <= 3 * ensemble_std_raw) * 100
    
    if verbose:
        print(f"\n{dataset_name} Evaluation Results:")
        print(f"  True value range: {y_raw.min():.4f} to {y_raw.max():.4f} ppb")
        print(f"  Predicted range: {ensemble_mean_raw.min():.4f} to {ensemble_mean_raw.max():.4f} ppb")
        print(f"  Uncertainty range: {ensemble_std_raw.min():.4f} to {ensemble_std_raw.max():.4f} ppb")
        print(f"  RMSE: {rmse:.4f} ppb")
        print(f"  R²: {r2:.4f}")
        print(f"  ME: {me:.4f} ppb")
        print(f"  MAE: {mae:.4f} ppb")
        print(f"  3σ Coverage: {within_3sigma:.2f}%")
    
    return {
        'predictions': ensemble_mean_raw,
        'uncertainties': ensemble_std_raw,
        'true_values': y_raw,
        'rmse': rmse,
        'r2': r2,
        'me': me,
        'mae': mae,
        'coverage_3sigma': within_3sigma
    }


def evaluate_by_year(
    models: List[Dict],
    data_df,
    feature_columns: List[str],
    target_column: str,
    device: torch.device,
    output_mean: float,
    output_scale: float,
    years: Optional[List[int]] = None
) -> Dict:
    """Evaluate model performance by year."""
    if years is None:
        years = sorted(data_df['year'].unique())
    
    results = {}
    
    for year in years:
        year_data = data_df[data_df['year'] == year]
        if len(year_data) == 0:
            continue
        
        X_year = torch.tensor(year_data[feature_columns].values, dtype=torch.float32).to(device)
        y_year = torch.tensor(year_data[target_column].values.reshape(-1, 1), dtype=torch.float32).to(device)
        
        ensemble_mean, ensemble_var = ensemble_predict(models, X_year, device)
        
        ensemble_mean_raw = ensemble_mean.flatten() * output_scale + output_mean
        y_raw = y_year.cpu().numpy().flatten() * output_scale + output_mean
        
        rmse = np.sqrt(mean_squared_error(y_raw, ensemble_mean_raw))
        r2 = r2_score(y_raw, ensemble_mean_raw)
        me = np.mean(ensemble_mean_raw - y_raw)
        mae = np.mean(np.abs(ensemble_mean_raw - y_raw))
        
        results[year] = {
            'n_samples': len(year_data),
            'rmse': rmse,
            'r2': r2,
            'me': me,
            'mae': mae
        }
    
    return results

