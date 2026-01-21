import torch
import torch.nn as nn
import numpy as np


class ProbabilisticMLP(nn.Module):
    def __init__(self, input_size):
        super(ProbabilisticMLP, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.mean_head = nn.Linear(64, 1)
        self.variance_head = nn.Linear(64, 1)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        mean = self.mean_head(shared_features)
        variance = torch.nn.functional.softplus(self.variance_head(shared_features)) + 1e-6
        return mean, variance


def gaussian_nll_loss(mean, variance, target):
    """Gaussian Negative Log-Likelihood Loss."""
    loss = 0.5 * torch.log(variance) + (target - mean) ** 2 / (2 * variance)
    return torch.mean(loss)


def ensemble_predict(models, X, device):
    """Ensemble prediction using multiple model snapshots."""
    predictions_mean = []
    predictions_variance = []
    
    for model_state in models:
        temp_model = ProbabilisticMLP(X.shape[1])
        temp_model.load_state_dict({k: v.to(device) for k, v in model_state.items()})
        temp_model.to(device)
        temp_model.eval()
        
        with torch.no_grad():
            mean, variance = temp_model(X)
            predictions_mean.append(mean.cpu().numpy())
            predictions_variance.append(variance.cpu().numpy())
    
    predictions_mean = np.array(predictions_mean)
    predictions_variance = np.array(predictions_variance)
    M = len(predictions_mean)
    
    ensemble_mean = np.sum(predictions_mean, axis=0) / M
    first_term = np.sum(predictions_variance, axis=0) / M
    second_term = np.sum((predictions_mean - ensemble_mean) ** 2, axis=0) / M
    ensemble_variance = first_term + second_term
    
    return ensemble_mean, ensemble_variance
