"""
dashboard.py – Generate an interactive HTML dashboard for experiment results.

Reads experiment_results.json and produces a standalone HTML file
with charts for training progress, privacy budget, and attack metrics.
"""

import json
import os
import sys


def generate_dashboard(results_path='results/experiment_results.json',
                       output_path='results/dashboard.html'):
    """Generate interactive HTML dashboard from experiment results."""

    with open(results_path, 'r') as f:
        results = json.load(f)

    history = results['history']
    config = results['config']
    attack = results.get('attack_results')
    label_dist = results.get('label_distribution', [])

    rounds_json = json.dumps(history['rounds'])
    accuracy_json = json.dumps(history['test_accuracy'])
    loss_json = json.dumps(history['test_loss'])
    epsilon_json = json.dumps(history['epsilon'])
    label_dist_json = json.dumps(label_dist)

    attack_auc = attack['auc'] if attack else 'N/A'
    attack_acc = attack['accuracy'] if attack else 'N/A'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FL + DP Experiment Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  :root {{
    --bg: #0a0e17;
    --surface: #111827;
    --border: #1e293b;
    --accent: #22d3ee;
    --accent-dim: #0e7490;
    --warning: #f59e0b;
    --success: #10b981;
    --danger: #ef4444;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --text-bright: #f8fafc;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 2rem;
  }}

  .header {{
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}

  .header h1 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-bright);
    letter-spacing: -0.02em;
  }}

  .header h1 span {{
    color: var(--accent);
  }}

  .header p {{
    color: var(--text-dim);
    font-size: 0.9rem;
    margin-top: 0.5rem;
  }}

  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}

  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
  }}

  .metric-card .label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
  }}

  .metric-card .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-bright);
  }}

  .metric-card .value.accent {{ color: var(--accent); }}
  .metric-card .value.success {{ color: var(--success); }}
  .metric-card .value.warning {{ color: var(--warning); }}
  .metric-card .value.danger {{ color: var(--danger); }}

  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
  }}

  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
  }}

  .chart-card h3 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--accent);
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  .chart-card.full-width {{
    grid-column: 1 / -1;
  }}

  .config-table {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }}

  .config-table h3 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--accent);
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  .config-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 0.5rem;
  }}

  .config-item {{
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0.75rem;
    border-radius: 6px;
    font-size: 0.85rem;
  }}

  .config-item:nth-child(odd) {{ background: rgba(255,255,255,0.02); }}
  .config-item .key {{ color: var(--text-dim); }}
  .config-item .val {{
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-bright);
    font-weight: 600;
  }}

  @media (max-width: 768px) {{
    .charts-grid {{ grid-template-columns: 1fr; }}
    body {{ padding: 1rem; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>🔒 Federated Learning + <span>Differential Privacy</span></h1>
  <p>Experiment Dashboard — {config['num_clients']} clients, {config['num_rounds']} rounds, σ={config['noise_multiplier']}</p>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="label">Final Test Accuracy</div>
    <div class="value success">{history['test_accuracy'][-1]:.1%}</div>
  </div>
  <div class="metric-card">
    <div class="label">Final ε (δ={config['target_delta']})</div>
    <div class="value accent">{history['epsilon'][-1]:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="label">MIA AUC</div>
    <div class="value {'success' if attack and attack['auc'] <= 0.55 else 'danger'}">{f"{attack_auc:.4f}" if isinstance(attack_auc, float) else attack_auc}</div>
  </div>
  <div class="metric-card">
    <div class="label">Training Rounds</div>
    <div class="value">{config['num_rounds']}</div>
  </div>
  <div class="metric-card">
    <div class="label">Noise Multiplier σ</div>
    <div class="value warning">{config['noise_multiplier']}</div>
  </div>
</div>

<div class="charts-grid">
  <div class="chart-card">
    <h3>Test Accuracy per Round</h3>
    <canvas id="accuracyChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Privacy Budget (ε) Accumulation</h3>
    <canvas id="epsilonChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Test Loss per Round</h3>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Client Data Distribution (Non-IID)</h3>
    <canvas id="distChart"></canvas>
  </div>
</div>

<div class="config-table">
  <h3>Experiment Configuration</h3>
  <div class="config-grid">
    <div class="config-item"><span class="key">Clients</span><span class="val">{config['num_clients']}</span></div>
    <div class="config-item"><span class="key">Rounds</span><span class="val">{config['num_rounds']}</span></div>
    <div class="config-item"><span class="key">Local Epochs</span><span class="val">{config['local_epochs']}</span></div>
    <div class="config-item"><span class="key">Batch Size</span><span class="val">{config['batch_size']}</span></div>
    <div class="config-item"><span class="key">Learning Rate</span><span class="val">{config['lr']}</span></div>
    <div class="config-item"><span class="key">DP Enabled</span><span class="val">{config['use_dp']}</span></div>
    <div class="config-item"><span class="key">Noise Multiplier (σ)</span><span class="val">{config['noise_multiplier']}</span></div>
    <div class="config-item"><span class="key">Max Grad Norm (C)</span><span class="val">{config['max_grad_norm']}</span></div>
    <div class="config-item"><span class="key">Target δ</span><span class="val">{config['target_delta']}</span></div>
    <div class="config-item"><span class="key">Non-IID α</span><span class="val">{config['alpha']}</span></div>
  </div>
</div>

<script>
const chartDefaults = {{
  color: '#94a3b8',
  borderColor: '#1e293b',
  font: {{ family: 'IBM Plex Sans' }}
}};
Chart.defaults.color = chartDefaults.color;
Chart.defaults.font.family = chartDefaults.font.family;

const rounds = {rounds_json};
const accuracy = {accuracy_json};
const loss = {loss_json};
const epsilon = {epsilon_json};
const labelDist = {label_dist_json};

// Accuracy Chart
new Chart(document.getElementById('accuracyChart'), {{
  type: 'line',
  data: {{
    labels: rounds,
    datasets: [{{
      label: 'Test Accuracy',
      data: accuracy,
      borderColor: '#10b981',
      backgroundColor: 'rgba(16, 185, 129, 0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 2,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Round' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ title: {{ display: true, text: 'Accuracy' }}, grid: {{ color: '#1e293b' }}, min: 0, max: 1 }}
    }}
  }}
}});

// Epsilon Chart
new Chart(document.getElementById('epsilonChart'), {{
  type: 'line',
  data: {{
    labels: rounds,
    datasets: [{{
      label: 'ε (epsilon)',
      data: epsilon,
      borderColor: '#22d3ee',
      backgroundColor: 'rgba(34, 211, 238, 0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 2,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Round' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ title: {{ display: true, text: 'ε' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Loss Chart
new Chart(document.getElementById('lossChart'), {{
  type: 'line',
  data: {{
    labels: rounds,
    datasets: [{{
      label: 'Test Loss',
      data: loss,
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245, 158, 11, 0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 2,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Round' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ title: {{ display: true, text: 'Loss' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Data Distribution Chart
if (labelDist.length > 0) {{
  const colors = ['#ef4444','#f59e0b','#10b981','#22d3ee','#8b5cf6','#ec4899','#f97316','#14b8a6','#6366f1','#a855f7'];
  const datasets = [];
  for (let cls = 0; cls < 10; cls++) {{
    datasets.push({{
      label: `Digit ${{cls}}`,
      data: labelDist.map(row => row[cls]),
      backgroundColor: colors[cls],
    }});
  }}
  new Chart(document.getElementById('distChart'), {{
    type: 'bar',
    data: {{
      labels: labelDist.map((_, i) => `Client ${{i}}`),
      datasets: datasets,
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: 'right', labels: {{ boxWidth: 12, padding: 6, font: {{ size: 10 }} }} }} }},
      scales: {{
        x: {{ stacked: true, grid: {{ color: '#1e293b' }} }},
        y: {{ stacked: true, title: {{ display: true, text: 'Samples' }}, grid: {{ color: '#1e293b' }} }}
      }}
    }}
  }});
}}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Dashboard saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    generate_dashboard()
