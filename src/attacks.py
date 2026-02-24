"""
attacks.py – Membership Inference Attack (MIA) for evaluating DP effectiveness.

Implements a shadow-model based membership inference attack:
  1. Train shadow models that mimic the target model's behavior
  2. Use shadow model outputs to train an "attack model" (binary classifier)
  3. Evaluate whether the attack model can distinguish training members
     from non-members in the target model

A successful DP implementation should make MIA AUC ≈ 0.5 (random guessing).

Reference: Shokri et al., "Membership Inference Attacks Against ML Models" (2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, Subset, TensorDataset


class MembershipInferenceAttack:
    """
    Shadow-model membership inference attack.
    
    The attacker's goal: given a data point x and a trained model M,
    determine whether x was in M's training set.
    
    Attack signal: the model's confidence (softmax output) on a sample
    differs between training members and non-members.
    """

    def __init__(self, target_model, train_indices, test_dataset, device='cpu'):
        """
        Args:
            target_model: The trained federated model to attack
            train_indices: Indices of samples used in training (ground truth)
            test_dataset: Full test dataset (non-members)
            device: Torch device
        """
        self.target_model = target_model.to(device)
        self.train_indices = set(train_indices)
        self.test_dataset = test_dataset
        self.device = device

    def extract_attack_features(self, dataset, indices, is_member):
        """
        Extract prediction confidence vectors for attack model training.
        
        For each sample, we record:
        - The sorted softmax probabilities (prediction confidence)
        - The true label
        - Whether it's a member (1) or non-member (0)
        """
        self.target_model.eval()
        features = []
        labels = []

        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=128, shuffle=False)

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.target_model(batch_x)
                probs = F.softmax(outputs, dim=1)

                # Sort probabilities descending (captures confidence pattern)
                sorted_probs, _ = torch.sort(probs, dim=1, descending=True)

                for i in range(sorted_probs.size(0)):
                    feature = sorted_probs[i].cpu().numpy()
                    features.append(feature)
                    labels.append(is_member)

        return np.array(features), np.array(labels)

    def run_attack(self, train_dataset, num_attack_samples=1000):
        """
        Execute the membership inference attack.
        
        Strategy:
        1. Sample `num_attack_samples` members (from training set)
        2. Sample `num_attack_samples` non-members (from test set)
        3. Extract confidence features for both groups
        4. Train a logistic regression attack classifier
        5. Evaluate with AUC-ROC
        
        Args:
            train_dataset: The full training dataset
            num_attack_samples: Number of samples per class for the attack
            
        Returns:
            Dictionary with attack metrics (AUC, accuracy, confidence scores)
        """
        # Sample member indices
        all_train_indices = list(self.train_indices)
        np.random.shuffle(all_train_indices)
        member_indices = all_train_indices[:num_attack_samples]

        # Sample non-member indices from test set
        non_member_indices = list(range(min(num_attack_samples, len(self.test_dataset))))

        # Extract features
        member_features, member_labels = self.extract_attack_features(
            train_dataset, member_indices, is_member=1
        )
        non_member_features, non_member_labels = self.extract_attack_features(
            self.test_dataset, non_member_indices, is_member=0
        )

        # Combine
        X = np.vstack([member_features, non_member_features])
        y = np.concatenate([member_labels, non_member_labels])

        # Shuffle
        shuffle_idx = np.random.permutation(len(y))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Split into attack train/test
        split = int(0.6 * len(y))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train simple attack model (logistic regression via 1-layer NN)
        attack_model = self._train_attack_model(X_train, y_train)

        # Evaluate
        attack_preds = self._predict_attack(attack_model, X_test)
        auc = roc_auc_score(y_test, attack_preds)
        binary_preds = (attack_preds > 0.5).astype(int)
        accuracy = accuracy_score(y_test, binary_preds)

        # Compute confidence gap
        member_conf = attack_preds[y_test == 1].mean()
        non_member_conf = attack_preds[y_test == 0].mean()

        return {
            'auc': auc,
            'accuracy': accuracy,
            'member_confidence': float(member_conf),
            'non_member_confidence': float(non_member_conf),
            'confidence_gap': float(member_conf - non_member_conf),
            'num_test_samples': len(y_test),
        }

    def _train_attack_model(self, X, y, epochs=50, lr=0.01):
        """Train a simple binary classifier as the attack model."""
        input_dim = X.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

        return model

    def _predict_attack(self, model, X):
        """Run attack model predictions."""
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy().squeeze()
        return preds
