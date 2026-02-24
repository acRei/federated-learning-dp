"""
main.py – Orchestrator for the Federated Learning + Differential Privacy pipeline.

Runs the full experiment:
  1. Load & partition MNIST into non-IID client splits
  2. Initialize global model and federated server
  3. Run federated training rounds with DP-SGD
  4. Track privacy budget (ε) via RDP accountant
  5. Evaluate model accuracy and run membership inference attack
  6. Save results and generate visualizations
"""

import argparse
import json
import os
import sys
import time
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MNISTNet
from src.data import get_mnist, partition_non_iid, create_client_loaders, get_label_distribution
from src.client import FederatedClient
from src.server import FederationServer
from src.privacy import RDPAccountant, calibrate_noise
from src.attacks import MembershipInferenceAttack


def parse_args():
    parser = argparse.ArgumentParser(
        description='Federated Learning with Differential Privacy'
    )

    # Federated learning parameters
    parser.add_argument('--num_clients', type=int, default=5,
                        help='Number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=30,
                        help='Number of federated training rounds')
    parser.add_argument('--local_epochs', type=int, default=3,
                        help='Local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Local batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Local learning rate')

    # Differential privacy parameters
    parser.add_argument('--use_dp', action='store_true', default=True,
                        help='Enable differential privacy')
    parser.add_argument('--noise_multiplier', type=float, default=0.8,
                        help='DP noise multiplier σ')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='DP gradient clipping threshold C')
    parser.add_argument('--target_delta', type=float, default=1e-5,
                        help='Target δ for (ε, δ)-DP')

    # Data parameters
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet α for non-IID partitioning')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Attack parameters
    parser.add_argument('--run_attack', action='store_true', default=True,
                        help='Run membership inference attack')
    parser.add_argument('--num_attack_samples', type=int, default=500,
                        help='Samples per class for MIA')

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Federated Learning with Differential Privacy")
    print(f"{'='*60}")
    print(f"  Device:           {device}")
    print(f"  Clients:          {args.num_clients}")
    print(f"  Rounds:           {args.num_rounds}")
    print(f"  Local epochs:     {args.local_epochs}")
    print(f"  DP enabled:       {args.use_dp}")
    print(f"  Noise multiplier: {args.noise_multiplier}")
    print(f"  Max grad norm:    {args.max_grad_norm}")
    print(f"  Target δ:         {args.target_delta}")
    print(f"  Non-IID α:        {args.alpha}")
    print(f"{'='*60}\n")

    # ── Step 1: Load and partition data ──────────────────────────
    print("[1/5] Loading MNIST and partitioning across clients...")
    train_dataset, test_dataset = get_mnist()
    client_indices = partition_non_iid(
        train_dataset, args.num_clients, alpha=args.alpha, seed=args.seed
    )

    # Print data distribution
    label_dist = get_label_distribution(train_dataset, client_indices)
    for i in range(args.num_clients):
        dominant_classes = np.argsort(label_dist[i])[-3:][::-1]
        print(f"  Client {i}: {len(client_indices[i]):,} samples | "
              f"dominant classes: {dominant_classes.tolist()}")

    client_loaders = create_client_loaders(
        train_dataset, client_indices, batch_size=args.batch_size
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # ── Step 2: Initialize model and server ──────────────────────
    print(f"\n[2/5] Initializing model ({MNISTNet().count_parameters():,} parameters)...")
    global_model = MNISTNet()
    server = FederationServer(global_model, test_loader, device)

    # Initialize clients
    clients = [
        FederatedClient(i, loader, device)
        for i, loader in enumerate(client_loaders)
    ]

    # Initialize privacy accountant
    accountant = RDPAccountant()
    # sample_rate = probability each record appears in a batch (q = batch_size / dataset_size)
    min_client_size = min(len(indices) for indices in client_indices)
    sample_rate = min(args.batch_size / min_client_size, 1.0)

    # ── Step 3: Federated training ───────────────────────────────
    print(f"\n[3/5] Starting federated training ({args.num_rounds} rounds)...\n")

    history = {
        'rounds': [],
        'test_accuracy': [],
        'test_loss': [],
        'epsilon': [],
        'avg_client_loss': [],
        'avg_client_accuracy': [],
    }

    for round_num in range(1, args.num_rounds + 1):
        round_start = time.time()

        # Collect updates from all clients
        client_deltas = []
        client_samples = []
        client_metrics_list = []

        for client in clients:
            delta, metrics = client.local_train(
                global_model=server.get_global_model(),
                local_epochs=args.local_epochs,
                lr=args.lr,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
                use_dp=args.use_dp,
            )
            client_deltas.append(delta)
            client_samples.append(metrics['num_samples'])
            client_metrics_list.append(metrics)

        # Aggregate updates
        server.aggregate(client_deltas, client_samples)

        # Track privacy budget
        if args.use_dp:
            # Each client performs local_epochs * ceil(n_samples/batch_size) steps
            steps_per_round = args.local_epochs * max(
                len(loader) for loader in client_loaders
            )
            for _ in range(steps_per_round):
                accountant.step(args.noise_multiplier, sample_rate)
            current_epsilon = accountant.get_epsilon(args.target_delta)
        else:
            current_epsilon = float('inf')

        # Evaluate global model
        test_metrics = server.evaluate()
        round_time = time.time() - round_start

        # Log
        avg_loss = np.mean([m['loss'] for m in client_metrics_list])
        avg_acc = np.mean([m['accuracy'] for m in client_metrics_list])

        history['rounds'].append(round_num)
        history['test_accuracy'].append(test_metrics['test_accuracy'])
        history['test_loss'].append(test_metrics['test_loss'])
        history['epsilon'].append(current_epsilon)
        history['avg_client_loss'].append(avg_loss)
        history['avg_client_accuracy'].append(avg_acc)

        eps_str = f"ε={current_epsilon:.2f}" if current_epsilon != float('inf') else "ε=∞"
        print(f"  Round {round_num:3d}/{args.num_rounds} | "
              f"Test Acc: {test_metrics['test_accuracy']:.4f} | "
              f"Loss: {test_metrics['test_loss']:.4f} | "
              f"{eps_str} | "
              f"Time: {round_time:.1f}s")

    # ── Step 4: Membership Inference Attack ──────────────────────
    if args.run_attack:
        print(f"\n[4/5] Running Membership Inference Attack...")
        all_train_indices = [idx for indices in client_indices for idx in indices]

        attack = MembershipInferenceAttack(
            target_model=server.get_global_model(),
            train_indices=all_train_indices,
            test_dataset=test_dataset,
            device=device,
        )
        attack_results = attack.run_attack(
            train_dataset,
            num_attack_samples=args.num_attack_samples
        )

        print(f"  Attack AUC:            {attack_results['auc']:.4f} "
              f"({'⚠ Vulnerable' if attack_results['auc'] > 0.55 else '✓ Protected'})")
        print(f"  Attack Accuracy:       {attack_results['accuracy']:.4f}")
        print(f"  Member Confidence:     {attack_results['member_confidence']:.4f}")
        print(f"  Non-member Confidence: {attack_results['non_member_confidence']:.4f}")
        print(f"  Confidence Gap:        {attack_results['confidence_gap']:.4f}")
    else:
        attack_results = None
        print("\n[4/5] Skipping membership inference attack.")

    # ── Step 5: Save results ─────────────────────────────────────
    print(f"\n[5/5] Saving results...")

    results = {
        'config': vars(args),
        'history': history,
        'final_metrics': {
            'test_accuracy': history['test_accuracy'][-1],
            'final_epsilon': history['epsilon'][-1],
        },
        'attack_results': attack_results,
        'label_distribution': label_dist.tolist(),
    }

    os.makedirs('results', exist_ok=True)
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Final Test Accuracy: {history['test_accuracy'][-1]:.4f}")
    if args.use_dp:
        print(f"  Final Privacy Budget: ε = {history['epsilon'][-1]:.2f} "
              f"(δ = {args.target_delta})")
    if attack_results:
        print(f"  MIA AUC: {attack_results['auc']:.4f}")
    print(f"  Results saved to: results/experiment_results.json")
    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    main()