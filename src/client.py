"""
client.py – Federated learning client with DP-SGD.

Each client trains locally on its private data partition, then
applies differential privacy mechanisms (gradient clipping + noise)
before sending updates to the server.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from src.privacy import clip_gradients, add_noise


class FederatedClient:
    """
    A federated learning participant with local DP-SGD training.
    
    Privacy is enforced at the client level:
    1. Each mini-batch gradient is clipped to bound sensitivity
    2. Calibrated Gaussian noise is added to clipped gradients
    3. Only the noised model update (Δw) is sent to the server
    
    The raw data never leaves the client.
    """

    def __init__(self, client_id, data_loader, device='cpu'):
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.num_samples = len(data_loader.dataset)

    def local_train(self, global_model, local_epochs, lr,
                    noise_multiplier=0.0, max_grad_norm=1.0,
                    use_dp=True):
        """
        Train the model locally and return the update delta.
        
        Args:
            global_model: Current global model (will be copied, not modified)
            local_epochs: Number of local training epochs
            lr: Learning rate
            noise_multiplier: DP noise parameter σ
            max_grad_norm: DP gradient clipping threshold C
            use_dp: Whether to apply DP mechanisms
            
        Returns:
            model_delta: Dictionary of parameter deltas (local - global)
            metrics: Dictionary with loss, accuracy, grad_norms
        """
        # Copy global model for local training
        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        # Save initial parameters to compute delta later
        initial_params = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        num_batches = 0

        for epoch in range(local_epochs):
            for batch_x, batch_y in self.data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()

                if use_dp:
                    # Step 1: Clip gradients to bound sensitivity
                    grad_norm = clip_gradients(model.parameters(), max_grad_norm)
                    grad_norms.append(grad_norm)

                    # Step 2: Add calibrated Gaussian noise
                    # Scale noise by 1/batch_size: in DP-SGD, noise is added
                    # to the *sum* of clipped gradients, then divided by batch size.
                    # Since PyTorch's CrossEntropyLoss averages over the batch,
                    # we scale the noise down accordingly.
                    batch_count = batch_x.size(0)
                    add_noise(
                        model.parameters(),
                        noise_multiplier=noise_multiplier,
                        max_grad_norm=max_grad_norm / batch_count,
                        device=self.device
                    )

                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)
                num_batches += 1

        # Compute model delta: Δw = w_local - w_global
        model_delta = {}
        for name, param in model.named_parameters():
            model_delta[name] = param.data - initial_params[name]

        metrics = {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': correct / max(total, 1),
            'avg_grad_norm': sum(grad_norms) / max(len(grad_norms), 1) if grad_norms else 0,
            'num_samples': self.num_samples,
        }

        return model_delta, metrics

    def __repr__(self):
        return f"FederatedClient(id={self.client_id}, samples={self.num_samples})"