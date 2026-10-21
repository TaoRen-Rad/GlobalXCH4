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
        self.log_sigma_head = nn.Linear(64, 1)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        mean = self.mean_head(shared_features)
        log_sigma = self.log_sigma_head(shared_features)
        return mean, log_sigma


def gaussian_nll_loss(mean, log_sigma, target):
    """Gaussian NLL with the predicted log standard deviation."""
    loss = log_sigma + 0.5 * (target - mean) ** 2 * torch.exp(-2 * log_sigma)
    return torch.mean(loss)


def ensemble_predict(models, X, device, batch_size=4096):
    """Return ensemble mean and variance, transferring one batch at a time."""
    X_cpu = X.detach().cpu()
    predictions_mean = []
    predictions_variance = []
    
    for model_state in models:
        temp_model = ProbabilisticMLP(X.shape[1])
        temp_model.load_state_dict({k: v.cpu() for k, v in model_state.items()})
        temp_model.to(device)
        temp_model.eval()
        
        means = []
        variances = []
        with torch.no_grad():
            for start in range(0, len(X_cpu), batch_size):
                batch = X_cpu[start:start + batch_size].to(device)
                mean, log_sigma = temp_model(batch)
                means.append(mean.cpu().numpy())
                variances.append(torch.exp(2 * log_sigma).cpu().numpy())
        predictions_mean.append(np.concatenate(means, axis=0))
        predictions_variance.append(np.concatenate(variances, axis=0))
        del temp_model
    
    predictions_mean = np.array(predictions_mean)
    predictions_variance = np.array(predictions_variance)
    M = len(predictions_mean)
    
    ensemble_mean = np.sum(predictions_mean, axis=0) / M
    first_term = np.sum(predictions_variance, axis=0) / M
    second_term = np.sum((predictions_mean - ensemble_mean) ** 2, axis=0) / M
    ensemble_variance = first_term + second_term
    
    return ensemble_mean, ensemble_variance
