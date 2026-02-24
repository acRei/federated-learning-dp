# 🔒 Federated Learning with Differential Privacy

A privacy-preserving machine learning pipeline that trains models across decentralized data sources without exposing raw data. Implements **Federated Averaging (FedAvg)** with **Differential Privacy (DP)** guarantees using gradient clipping and calibrated Gaussian noise injection.

##  Project Overview

This project demonstrates how organizations can collaboratively train ML models while:
- **Never sharing raw data** between participants (federated learning)
- **Bounding information leakage** from model updates (differential privacy)
- **Tracking privacy budgets** (ε, δ) across training rounds
- **Quantifying the privacy–utility tradeoff** empirically

### Architecture

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Client 1   │  │   Client 2   │  │   Client N   │
│  Local Data  │  │  Local Data  │  │  Local Data  │
│  Local Train │  │  Local Train │  │  Local Train │
│  + DP Noise  │  │  + DP Noise  │  │  + DP Noise  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │    Noised Gradients Only          │
       ▼                 ▼                 ▼
    ┌─────────────────────────────────────────┐
    │           Aggregation Server            │
    │  FedAvg: weighted average of updates    │
    │  Privacy Accountant: track (ε, δ)       │
    │  Global Model Update                    │
    └─────────────────────────────────────────┘
```

##  Key Concepts

| Concept | Implementation |
|---|---|
| **Federated Averaging** | Weighted aggregation of client model updates |
| **DP-SGD** | Per-sample gradient clipping + Gaussian noise |
| **Privacy Accountant** | Rényi Differential Privacy (RDP) → (ε, δ) conversion |
| **Non-IID Data** | Dirichlet-based data partitioning across clients |
| **Membership Inference Attack** | Shadow model attack to empirically validate DP protection |

##  Installation

```bash
# Clone the repository
git clone https://github.com/acRei/federated-learning-dp.git
cd federated-learning-dp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

##  Start

### 1. Run the Full Pipeline

```bash
python src/main.py
```

This will:
- Partition MNIST across 5 simulated clients (non-IID)
- Train a CNN using Federated Averaging + DP-SGD
- Track privacy budget (ε) consumption per round
- Output accuracy vs. privacy tradeoff metrics
- Run a membership inference attack to validate DP protection

### 2. Customize Parameters

```bash
python src/main.py \
  --num_clients 10 \
  --num_rounds 50 \
  --local_epochs 5 \
  --noise_multiplier 1.0 \
  --max_grad_norm 1.0 \
  --target_epsilon 8.0 \
  --target_delta 1e-5
```

### 3. Launch the Dashboard

```bash
python src/dashboard.py
```

Opens an interactive visualization showing training progress, per-round ε accumulation, and attack success rates.

## 📁 Project Structure

```
federated-learning-dp/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py              # Entry point & orchestrator
│   ├── model.py             # CNN architecture
│   ├── client.py            # Federated client with DP-SGD
│   ├── server.py            # Aggregation server + FedAvg
│   ├── privacy.py           # RDP accountant & noise calibration
│   ├── data.py              # Non-IID data partitioning
│   ├── attacks.py           # Membership inference attack
│   └── dashboard.py         # Results visualization (HTML)
├── tests/
│   └── test_privacy.py      # Unit tests for privacy guarantees
└── notebooks/
    └── analysis.ipynb        # Experiment analysis notebook
```

##  Results

Typical results on MNIST with 5 clients, 30 rounds:

| Configuration | Test Accuracy | Final ε (δ=1e-5) | MIA AUC |
|---|---|---|---|
| No DP (baseline) | 98.2% | ∞ | 0.62 |
| DP (σ=0.5) | 96.8% | 12.4 | 0.54 |
| DP (σ=1.0) | 94.1% | 4.7 | 0.51 |
| DP (σ=2.0) | 88.3% | 1.9 | 0.50 |

> **MIA AUC = 0.50** means the attacker performs no better than random guessing → strong privacy.

##  Key Takeaways

1. **Differential privacy provably bounds information leakage** — even a well-resourced attacker cannot reliably determine if a specific sample was in the training set.
2. **Privacy comes at a utility cost** — higher noise (lower ε) reduces accuracy, requiring careful calibration.
3. **Federated learning alone is not sufficient** — without DP, gradient updates can leak sensitive information via model inversion or membership inference attacks.
4. **RDP composition provides tighter bounds** than naïve composition, enabling more training rounds within a privacy budget.

##  References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (2017)
- Abadi et al., *Deep Learning with Differential Privacy* (2016)
- Mironov, *Rényi Differential Privacy* (2017)
- Shokri et al., *Membership Inference Attacks Against Machine Learning Models* (2017)

## License

MIT
