"""
server.py – Federated aggregation server.

Implements Federated Averaging (FedAvg): collects model updates from
clients and computes a weighted average to update the global model.

The server never sees raw data — only (noised) parameter deltas.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FederationServer:
    """
    Central aggregation server for federated learning.
    
    Responsibilities:
    - Maintain the global model
    - Aggregate client updates via weighted averaging
    - Evaluate global model on a held-out test set
    - Track training metrics across rounds
    """

    def __init__(self, global_model, test_loader, device='cpu'):
        self.global_model = global_model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.round_metrics = []

    def aggregate(self, client_deltas, client_num_samples):
        """
        Federated Averaging: weighted average of client model deltas.
        
        w_global += Σ (n_k / n_total) · Δw_k
        
        where n_k is the number of samples at client k.
        
        Args:
            client_deltas: List of parameter delta dicts from clients
            client_num_samples: List of sample counts per client
        """
        total_samples = sum(client_num_samples)

        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                # Weighted average of deltas
                weighted_delta = torch.zeros_like(param)
                for delta, n_samples in zip(client_deltas, client_num_samples):
                    weight = n_samples / total_samples
                    weighted_delta += weight * delta[name].to(self.device)

                # Apply aggregated update
                param.add_(weighted_delta)

    def evaluate(self):
        """
        Evaluate the global model on the test set.
        
        Returns:
            Dictionary with test loss and accuracy
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.global_model(batch_x)
                loss = criterion(outputs, batch_y)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)

        metrics = {
            'test_loss': test_loss / len(self.test_loader),
            'test_accuracy': correct / total,
        }
        self.round_metrics.append(metrics)
        return metrics

    def get_global_model(self):
        """Return a copy of the current global model."""
        return copy.deepcopy(self.global_model)

    def get_training_history(self):
        """Return all round metrics."""
        return self.round_metrics
