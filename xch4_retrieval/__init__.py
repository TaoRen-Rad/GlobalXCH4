"""
CH4 Fast Inversion with Fine-tuning
A probabilistic MLP ensemble model for CH4 retrieval from TROPOMI and GOSAT satellite data.
"""

from .model import ProbabilisticMLP, gaussian_nll_loss, ensemble_predict
from .process_data import (
    generate_scalers_from_raw_data,
    standardize_data
)
from .train import train_model, save_model, load_model
from .finetune import finetune_model
from .evaluate import evaluate_model, evaluate_by_year
from .extract_data import (
    get_valid_points,
    extract_l1b_params,
    process_l1b_l2_pair
)

__version__ = "1.0.0"
__all__ = [
    "ProbabilisticMLP",
    "gaussian_nll_loss",
    "ensemble_predict",
    "generate_scalers_from_raw_data",
    "standardize_data",
    "train_model",
    "save_model",
    "load_model",
    "finetune_model",
    "evaluate_model",
    "evaluate_by_year",
    "get_valid_points",
    "extract_l1b_params",
    "process_l1b_l2_pair",
]

