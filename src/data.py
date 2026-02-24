"""
data.py – Non-IID data partitioning for federated learning.

Simulates realistic federated scenarios where each client has a
different data distribution (non-IID). Uses Dirichlet allocation
to control the degree of heterogeneity.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist(data_dir='./data'):
    """Download and return MNIST train/test datasets with normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def partition_non_iid(dataset, num_clients, alpha=0.5, seed=42):
    """
    Partition dataset into non-IID splits using Dirichlet distribution.
    
    The Dirichlet parameter α controls heterogeneity:
      - α → 0: Each client gets data from ~1 class (extremely non-IID)
      - α → ∞: Each client gets a uniform sample (IID)
      - α = 0.5: Moderate heterogeneity (realistic scenario)
    
    Args:
        dataset: PyTorch dataset with .targets attribute
        num_clients: Number of federated clients
        alpha: Dirichlet concentration parameter
        seed: Random seed for reproducibility
        
    Returns:
        List of index arrays, one per client
    """
    np.random.seed(seed)

    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)

        # Draw proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to actual counts
        proportions = (proportions * len(class_indices)).astype(int)

        # Fix rounding: assign remaining samples to random clients
        remainder = len(class_indices) - proportions.sum()
        for i in range(abs(remainder)):
            idx = np.random.randint(0, num_clients)
            proportions[idx] += np.sign(remainder)

        # Split and assign
        start = 0
        for client_id in range(num_clients):
            end = start + proportions[client_id]
            client_indices[client_id].extend(class_indices[start:end].tolist())
            start = end

    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def create_client_loaders(dataset, client_indices, batch_size=64):
    """
    Create DataLoaders for each client from their assigned indices.
    
    Args:
        dataset: Full training dataset
        client_indices: List of index arrays from partition_non_iid
        batch_size: Batch size for local training
        
    Returns:
        List of DataLoaders, one per client
    """
    loaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    return loaders


def get_label_distribution(dataset, client_indices):
    """
    Compute the label distribution for each client (for visualization).
    
    Returns:
        numpy array of shape (num_clients, num_classes)
    """
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])

    num_classes = len(np.unique(labels))
    num_clients = len(client_indices)
    distribution = np.zeros((num_clients, num_classes))

    for client_id, indices in enumerate(client_indices):
        for idx in indices:
            distribution[client_id][labels[idx]] += 1

    return distribution
